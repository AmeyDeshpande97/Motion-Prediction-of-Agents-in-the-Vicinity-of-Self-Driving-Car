import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6
n_hidden = 2048

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--dev-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--img-res",
        type=int,
        required=False,
        default=IMG_RES,
        help="Input images resolution",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=IN_CHANNELS,
        help="Input raster channels",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        required=False,
        default=TL,
        help="Number time step to predict",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        required=False,
        default=N_TRAJS,
        help="Number of trajectories to predict",
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save model and logs"
    )

    parser.add_argument(
        "--model", type=str, required=False, default="xception71", help="CNN model name"
    )
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch-size", type=int, required=False, default=48)
    parser.add_argument("--n-epochs", type=int, required=False, default=60)

    parser.add_argument("--valid-limit", type=int, required=False, default=24 * 100)
    parser.add_argument(
        "--n-monitor-train",
        type=int,
        required=False,
        default=10,
        help="Validate model each `n-validate` steps",
    )
    parser.add_argument(
        "--n-monitor-validate",
        type=int,
        required=False,
        default=1000,
        help="Validate model each `n-validate` steps",
    )

    args = parser.parse_args()

    return args

model_name = 'xception71'
in_channels = 25
n_traj = 6
time_limit = 80

features = {}

# Extracting features from the CNN
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

xception71 = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=n_traj * 2 * time_limit + n_traj,
        )

# Extracting the features from the CNN
# xception71.blocks(22).act_pw.register_forward_hook(get_features('feats'))
xception71.head.global_pool.register_forward_hook(get_features('feats'))

class CNN_LSTM(nn.Module):
    def __init__(self, model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS, n_hidden = n_hidden):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit 
        self.n_hidden = n_hidden
        self.CNN = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
        )
        self.lstm = nn.LSTMCell(1,self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future = 80): 
        model = self.CNN(x) 
          
        predictions = []
        n = x.size(0) # Number of samples
        h_t = features
        c_t = torch.zeros(n, self.n_hidden, dtype=torch.double)

        for input_t in x.split(1, dim=1):
            # x = 10x1
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            predictions += [output]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            predictions += [output]
        predictions = torch.cat(predictions, dim=1)
        print("predictions: ", predictions.size())
        
        return predictions


# x = [10, 25, 224, 224]
# outputs = [10,966]
# my_outputs = [10,128]


def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):

    # gt = [10,1,80,2] with unsqueeze
    # logits = [10,6,80,2] with unsqueeze
    # confidences = [10,6]
    # avails = [10,1,80,1] with unsqueeze

    # error = [10, 6, 80] reducing coords
    # error logsum = [10, 1]    
    # mean error = float number

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    
    return torch.mean(error)


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        trajectory = data["gt_marginal"]
        # print("trajectory: ", trajectory.shape)

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"]

        return raster, trajectory, is_available


def main():
    args = parse_args()
    print("Hi")
    summary_writer = SummaryWriter(os.path.join(args.save, "logs"))

    train_path = args.train_data
    dev_path = args.dev_data
    path_to_save = args.save
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    dataset = WaymoLoader(train_path)


    batch_size = args.batch_size
    num_workers = min(16, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dataset = WaymoLoader(dev_path, limit=args.valid_limit)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )



    model_name = args.model
    time_limit = args.time_limit
    n_traj = args.n_traj
    n_hidden = 2048
    model = CNN_LSTM(
            model_name, in_channels=args.in_channels, time_limit=time_limit, n_traj=n_traj, n_hidden = n_hidden
          )
    model.cuda()
  
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(dataloader),
        T_mult=1,
        eta_min=max(1e-2 * lr, 1e-6),
        last_epoch=-1,
    )

    start_iter = 0
    best_loss = float("+inf")
    glosses = []

    tr_it = iter(dataloader)
    n_epochs = args.n_epochs
    progress_bar = tqdm(range(start_iter, len(dataloader) * n_epochs))

    saver = lambda name: torch.save(
        {
            "score": best_loss,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
        },
        os.path.join(path_to_save, name),
    )
    
    writer = SummaryWriter(f'runs/Loss_plot')
    step = 0
  
    print("progress_bar =", progress_bar)
    for iteration in progress_bar:
        
        
        try:
            x, y, is_available = next(tr_it)
            # print("x :", x.size()) # x: [10,25, 224, 224]
            # print("y :", y.size()) # y: [10, 80, 2]
            # print("is_available :", is_available.size()) # is_available: [10, 80]
        except StopIteration:
            tr_it = iter(dataloader)
            x, y, is_available = next(tr_it)

        preds = []
        feats = []
        
        my_model = xception71(x)
        # inputs = x.to(device)
        # preds = model(inputs)
        print("features: ", features['feats'].size())
        n_hidden = features['feats'].size(1)
        print("n_hidden: ", n_hidden) 
      

        # features = CNN(x) # [10,966]
        # features = torch.unsqueeze(features, 1) # [10, 1, 966]
        # #print("features :", features.size())
        # net_lstm = LSTM()
        # lstm_output = net_lstm(features) #[10,6]
        #print("lstm_outputs: ", lstm_output.size())
        x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

        
        # preds.append(preds.detach().cpu().numpy())
        # feats.append(features['feats'].cpu().numpy())

        model.train()
        
        optimizer.zero_grad()

        # confidences_logits, logits = model(x)
        lstm_output = model(x)

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        glosses.append(loss.item())
        if (iteration + 1) % args.n_monitor_train == 0:
            progress_bar.set_description(
                f"loss: {loss.item():.3}"
                f" avg: {np.mean(glosses[-100:]):.2}"
                f" {scheduler.get_last_lr()[-1]:.3}"
            )
            summary_writer.add_scalar("train/loss", loss.item(), iteration)
            summary_writer.add_scalar("lr", scheduler.get_last_lr()[-1], iteration)
        
                
        
        if (iteration + 1) % args.n_monitor_validate == 0:
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for x, y, is_available in val_dataloader:
                    x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

                    confidences_logits, logits = model(x)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        y, logits, confidences_logits, is_available
                    )
                    val_losses.append(loss.item())

                summary_writer.add_scalar("dev/loss", np.mean(val_losses), iteration)

            saver("model_last.pth")

            mean_val_loss = np.mean(val_losses)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                saver("model_best.pth")

                model.eval()
                with torch.no_grad():
                    traced_model = torch.jit.trace(
                        model,
                        torch.rand(
                            1, args.in_channels, args.img_res, args.img_res
                        ).cuda(),
                    )

                traced_model.save(os.path.join(path_to_save, "model_best.pt"))
                del traced_model
    print("glosses: ", len(glosses))

        

if __name__ == "__main__":
    main()
