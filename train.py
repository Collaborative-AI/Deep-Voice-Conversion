import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import CPCDataset_sameSeq as CPCDataset
from scheduler import WarmupScheduler
from models.model_encoder import ContentEncoder, CPCLoss_sameSeq, StyleEncoder
from models.model_encoder_contrastive import ASE
from models.cross_modal_decoder import CrossModalDecoder, get_mask_from_lengths
from models.model_decoder import Decoder_ac_without_lf0 as DecoderWaveGAN
# from models.contrastive_loss import NTXent
from models.mi_estimators import CLUBSample_group

# import apex.amp as amp
import os
import time

torch.manual_seed(137)
np.random.seed(137)

def save_checkpoint(encoder_content, cpc, encoder_style, model_ase, \
                    cs_mi_net, decoder, \
                    optimizer, optimizer_cs_mi_net, scheduler, epoch, checkpoint_dir, cfg):
    checkpoint_state = {
### Modified here
        "encoder_content": encoder_content.state_dict(),
        # "encoder_lf0": encoder_lf0.state_dict(),
        "model_ase": model_ase.state_dict(),
        "cpc": cpc.state_dict(),
        "encoder_style": encoder_style.state_dict(),
        # "ps_mi_net": ps_mi_net.state_dict(),
        # "cp_mi_net": cp_mi_net.state_dict(),
        "cs_mi_net": cs_mi_net.state_dict(), 
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_cs_mi_net": optimizer_cs_mi_net.state_dict(),
        # "optimizer_ps_mi_net": optimizer_ps_mi_net.state_dict(),
        # "optimizer_cp_mi_net": optimizer_cp_mi_net.state_dict(),
        "scheduler": scheduler.state_dict(),
        # "amp": amp_state_dict,
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))

def mi_first_forward(mels, encoder_content, encoder_style, cs_mi_net, optimizer_cs_mi_net, cfg):
    optimizer_cs_mi_net.zero_grad()  
    style_mask = get_mask_from_lengths(mels)
    z, _, _, _, _ = encoder_content(mels) # z: bz x 64 x 64; bz x time x frequency
    z = z.detach()
    style_embs = encoder_style(mels.transpose(1,2), style_mask).detach() # bz x 256 x 1
    
    # Whether to Optimize the Mutual Information Loss * 3
    if cfg.use_CSMI:
        lld_cs_loss = -cs_mi_net.loglikeli(style_embs, z)
        lld_cs_loss.backward()
        optimizer_cs_mi_net.step()
    else:
        lld_cs_loss = torch.tensor(0.)
    
    # if cfg.use_CPMI:
    #     lld_cp_loss = -cp_mi_net.loglikeli(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
    #     if cfg.use_amp:
    #         with amp.scale_loss(lld_cp_loss, optimizer_cp_mi_net) as slll:
    #             slll.backward()
    #     else:
    #         lld_cp_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(cp_mi_net.parameters(), 1)
    #     optimizer_cp_mi_net.step()
    # else:
    #     lld_cp_loss = torch.tensor(0.)
        
    # if cfg.use_PSMI:
    #     lld_ps_loss = -ps_mi_net.loglikeli(style_embs, lf0_embs)
    #     if cfg.use_amp:
    #         with amp.scale_loss(lld_ps_loss, optimizer_ps_mi_net) as sll:
    #             sll.backward()
    #     else:
    #         lld_ps_loss.backward()
    #     optimizer_ps_mi_net.step()
    # else:
    #     lld_ps_loss = torch.tensor(0.)
            
    return optimizer_cs_mi_net, lld_cs_loss

def mi_second_forward(mels, prompts, mels_id, encoder_content, cpc, encoder_style, model_ase, cs_mi_net, decoder, cfg, optimizer, scheduler):
    optimizer.zero_grad()
    # mels: bz x 80 x 128; bz x frequency x time
    
    style_mask = get_mask_from_lengths(mels)
     
    z, c, _, vq_loss, perplexity = encoder_content(mels) # z: bz x 128/2 x 64; bz x time x frequency
    print("Content Embedding Shape: ", z.shape)
    print("CPC Content Embedding Shape: ", c.shape)
    cpc_loss, accuracy = cpc(z, c)
    style_embs = encoder_style(mels.transpose(1,2), style_mask) # bz x 256 x 1
    print("Style Embedding Shape: ", style_embs.shape)
    
    # # decode the linguistic content from the content embedding to fix a distribution for the mutual information loss 
    # linguistic_content = vocoder_content()
    # content_loss = F.cross_entropy(linguistic_content.transpose(1,2), mels_id)
    
    
    # Map the Prompt Embedding and the Style Embedding to the same space
    print("Mapping in to the same space...")
    contrastive_loss, style_embs, prompt_embs = model_ase(style_embs, prompts, mels_id) # prompt_embs: bz x 256 x 1; style_embs: bz x 256 x 1
    print("Style Embedding Shape: ", style_embs.shape)
    print("Prompt Embedding Shape: ", prompt_embs.shape)

    # Cross-Modal Attention Shift between Prompt Embedding and Content Embedding    
    # Decode the audio with Content Embedding and Prompt Embedding
    recon_loss, pred_mels = decoder(z, prompt_embs, mels.transpose(1,2))
    
    loss = recon_loss + cpc_loss + vq_loss + contrastive_loss
    
    if cfg.use_CSMI:
        mi_cs_loss = cfg.mi_weight*cs_mi_net.mi_est(style_embs, z)
    else:
        mi_cs_loss = torch.tensor(0.).to(loss.device)
    
    # if cfg.use_CPMI:
    #     mi_cp_loss = cfg.mi_weight*cp_mi_net.mi_est(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
    # else:
    #     mi_cp_loss = torch.tensor(0.).to(loss.device)
        
    # if cfg.use_PSMI:
    #     mi_ps_loss = cfg.mi_weight*ps_mi_net.mi_est(style_embs, lf0_embs)
    # else:
    #     mi_ps_loss = torch.tensor(0.).to(loss.device)
    
    # loss = loss + mi_cs_loss + mi_ps_loss + mi_cp_loss
    
    loss += mi_cs_loss
    
    loss.backward()
        
    torch.nn.utils.clip_grad_norm_(model_ase.parameters(), 2)
    optimizer.step()
    return optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss


def calculate_eval_loss(mels, prompts, mels_id, \
                        encoder_content, cpc, \
                        encoder_style, model_ase, cs_mi_net,\
                        decoder, cfg):
    with torch.no_grad():
        z, c, z_beforeVQ, vq_loss, perplexity = encoder_content(mels)
        style_embs = encoder_style(mels)
        
        if cfg.use_CSMI:
            lld_cs_loss = -cs_mi_net.loglikeli(style_embs, z)
            mi_cs_loss = cfg.mi_weight*cs_mi_net.mi_est(style_embs, z)
        else:
            lld_cs_loss = torch.tensor(0.)
            mi_cs_loss = torch.tensor(0.)
        
        # z, c, z_beforeVQ, vq_loss, perplexity = encoder(mels)
        cpc_loss, accuracy = cpc(z, c)
        contrastive_loss, style_embs, prompt_embs = model_ase(style_embs, prompts, mels_id)
        recon_loss, pred_mels = decoder(z, prompt_embs, mels.transpose(1,2))
        
        # if cfg.use_CPMI:
        #     mi_cp_loss = cfg.mi_weight*cp_mi_net.mi_est(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
        #     lld_cp_loss = -cp_mi_net.loglikeli(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
        # else:
        #     mi_cp_loss = torch.tensor(0.)
        #     lld_cp_loss = torch.tensor(0.)
            
        # if cfg.use_PSMI:
        #     mi_ps_loss = cfg.mi_weight*ps_mi_net.mi_est(style_embs, lf0_embs)
        #     lld_ps_loss = -ps_mi_net.loglikeli(style_embs, lf0_embs)
        # else:
        #     mi_ps_loss = torch.tensor(0.)
        #     lld_ps_loss = torch.tensor(0.)
            
        return recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, contrastive_loss


def to_eval(all_models):
    for m in all_models:
        m.eval()
        
        
def to_train(all_models):
    for m in all_models:
        m.train()
        
        
def eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder_content, cpc, encoder_style, model_ase, cs_mi_net, decoder, cfg):
    stime = time.time()
    average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = average_contrastive_loss = 0 
    average_accuracies = np.zeros(cfg.training.n_prediction_steps)
    average_lld_cs_loss = average_mi_cs_loss = 0
    all_models = [encoder_content, cpc, encoder_style, model_ase, cs_mi_net, decoder]
    to_eval(all_models)
    
    # TO DO: Change the formart of valid_dalaloader to (mels, prompts, mels_id)
    for i, (mels, prompts, mels_id) in enumerate(valid_dataloader, 1):
        # lf0 = lf0.to(device)
        mels = mels.to(device) # (bs, 80, 128)
        mels_id = mels_id.to(device)
        recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, contrastive_loss = \
            calculate_eval_loss(mels, prompts, mels_id, \
                        encoder_content, cpc, \
                        encoder_style, model_ase, cs_mi_net, \
                        decoder, cfg)
       
        average_recon_loss += (recon_loss.item() - average_recon_loss) / i
        average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
        average_vq_loss += (vq_loss.item() - average_vq_loss) / i
        average_perplexity += (perplexity.item() - average_perplexity) / i
        average_accuracies += (np.array(accuracy) - average_accuracies) / i
        average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
        average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
        average_contrastive_loss += (contrastive_loss.item() - average_contrastive_loss) / i
        # average_lld_ps_loss += (lld_ps_loss.item() - average_lld_ps_loss) / i
        # average_mi_ps_loss += (mi_ps_loss.item() - average_mi_ps_loss) / i
        # average_lld_cp_loss += (lld_cp_loss.item() - average_lld_cp_loss) / i
        # average_mi_cp_loss += (mi_cp_loss.item() - average_mi_cp_loss) / i
        
    
    ctime = time.time()
    print("Eval | epoch:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, contrastive loss:{:.3f},used time:{:.3f}s"
          .format(epoch, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_contrastive_loss, ctime-stime))
    print(100 * average_accuracies)
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
    results_txt.write("Eval | epoch:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, contrastive loss:{:.3f}"
          .format(epoch, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_contrastive_loss)+'\n')
    results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies])+'\n')
    results_txt.close()
    
    to_train(all_models)
    
    
@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    # with open('config/contrastive_settings.yaml', 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # config = DotMap(config)
    
    cfg.checkpoint_dir = f'{cfg.checkpoint_dir}/useCSMI{cfg.use_CSMI}'
    
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # define model
    encoder_content = ContentEncoder(**cfg.model.encoder_content)
    cpc = CPCLoss_sameSeq(**cfg.model.cpc)
    encoder_style = StyleEncoder(cfg.model.encoder_style)
    model_ase = ASE(cfg.model.contrastive_model)
    cs_mi_net = CLUBSample_group(256, cfg.model.encoder_content.z_dim, 512)
    # decoder = CrossModalDecoder(cfg.model.cross_modal_decoder)
    decoder = DecoderWaveGAN(dim_neck=cfg.model.encoder_content.z_dim, use_l1_loss=True)
    
    encoder_content.to(device)
    cpc.to(device)
    encoder_style.to(device)
    model_ase.to(device)
    cs_mi_net.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder_content.parameters(), cpc.parameters(), encoder_style.parameters(), model_ase.parameters(), decoder.parameters()),
        lr=cfg.training.scheduler.initial_lr)
    optimizer_cs_mi_net = optim.Adam(cs_mi_net.parameters(), lr=cfg.mi_lr)
    
    root_path = Path(utils.to_absolute_path("data"))
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames, # 128
        mode='train')
    valid_dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames, # 128
        mode='valid')
    
    warmup_epochs = 2000 // (len(dataset)//cfg.training.batch_size)
    print('warmup_epochs:', warmup_epochs)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        initial_lr=cfg.training.scheduler.initial_lr,
        max_lr=cfg.training.scheduler.max_lr,
        milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size, # 256
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size, # 256
        shuffle=False,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)
    
    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder_content.load_state_dict(checkpoint["encoder_content"])
        cpc.load_state_dict(checkpoint["cpc"])
        encoder_style.load_state_dict(checkpoint["encoder_style"])
        model_ase.load_state_dict(checkpoint["model_ase"])
        cs_mi_net.load_state_dict(checkpoint["cs_mi_net"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_cs_mi_net.load_state_dict(checkpoint["optimizer_cs_mi_net"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1
    
    if os.path.exists(f'{str(checkpoint_dir)}/results.txt'):
        wmode = 'a'
    else:
        wmode = 'w'
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', wmode)
    results_txt.write('save training info...\n')
    results_txt.close()
    
    global_step = 0
    stime = time.time()
    for epoch in range(start_epoch, cfg.training.n_epochs + 1):
        average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = average_contrastive_loss = 0
        average_accuracies = np.zeros(cfg.training.n_prediction_steps)
        average_lld_cs_loss = average_mi_cs_loss = 0

        # TO DO: Change the formart of dalaloader to (mels, prompts, mels_id)
        for i, (mels, prompts, mels_ids) in enumerate(dataloader, 1):
            # lf0 = lf0.to(device)
            mels = mels.to(device) # (bs, 80, 128); (bs, frequency/mel_channels, time/frames)
            mels_ids = mels_ids.to(device)
            if cfg.use_CSMI:
                optimizer_cs_mi_net, lld_cs_loss= mi_first_forward(mels, encoder_content, encoder_style, model_ase, cs_mi_net, optimizer_cs_mi_net, cfg)
            else:
                lld_cs_loss = torch.tensor(0.)
                
            optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, contrastive_loss = mi_second_forward(mels, prompts, mels_ids, \
                                                                                                            encoder_content, cpc, \
                                                                                                            encoder_style, model_ase, cs_mi_net, \
                                                                                                            decoder, cfg, \
                                                                                                            optimizer, scheduler)
           
            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i
            average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
            average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
            average_contrastive_loss += (contrastive_loss.item() - average_contrastive_loss) / i
 
            
            
            ctime = time.time()
            print("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, contrastive_loss:{:.3f}, used time:{:.3f}s"
                  .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_contrastive_loss, ctime-stime))
            print(100 * average_accuracies)
            stime = time.time()
            global_step += 1
            # scheduler.step()
            
        results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
        results_txt.write("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3E}, contrastive_loss:{:.3f}, mi cs loss:{:.3E}"
              .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_contrastive_loss)+'\n')
        results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies])+'\n')
        results_txt.close()
        scheduler.step()        
        
        if epoch % cfg.training.log_interval == 0 and epoch != start_epoch:
            eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder_content, cpc, encoder_style, model_ase, cs_mi_net, decoder, cfg)

            ctime = time.time()
            print("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, contrastive_loss:{:.3f}, used time:{:.3f}s"
                  .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_contrastive_loss, ctime-stime))
            print(100 * average_accuracies)
            stime = time.time()
            
        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(encoder_content, cpc, encoder_style, model_ase, \
                            cs_mi_net, decoder, \
                            optimizer, optimizer_cs_mi_net, scheduler, epoch, checkpoint_dir, cfg)


if __name__ == "__main__":
     train_model()
