import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6


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


class Model(nn.Module):
    def __init__(
        self, model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS
    ):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
        )
        self.model.head.fc = nn.Linear(2048,2048)
        self.head_bc1 = nn.BatchNorm1d(2048,eps=0.001)
        self.head_act1 = nn.LeakyReLU()
        self.head_fc2 = nn.Linear(2070,2070)
        self.head_bc2 = nn.BatchNorm1d(2070,eps=0.001)
        self.head_act2 = nn.LeakyReLU()
        self.head_fc3 = nn.Linear(2070,4096)
        self.head_bc3 = nn.BatchNorm1d(4096,eps=0.001)
        self.head_act3 = nn.LeakyReLU()
        self.head_fc4 = nn.Linear(4096,4096)
        self.head_bc4 = nn.BatchNorm1d(4096,eps=0.001)
        self.head_act4 = nn.LeakyReLU()
        self.head_fc5 = nn.Linear(4096,2048)
        self.head_bc5 = nn.BatchNorm1d(2048,eps=0.001)
        self.head_act5 = nn.LeakyReLU()
        self.head_fc6 = nn.Linear(2048,2048)
        self.head_bc6 = nn.BatchNorm1d(2048,eps=0.001)
        self.head_act6 = nn.LeakyReLU()
        self.head_fc7 = nn.Linear(2048,1926)

    def forward(self, raterize_im, xy_vel):
        xcept_outputs = self.model(raterize_im)
        outputs = self.head_bc1(xcept_outputs)
        outputs = self.head_act1(outputs)
        concat_output = torch.cat((outputs, xy_vel), dim=1)
        outputs = self.head_fc2(concat_output)
        outputs = self.head_bc2(outputs)
        outputs = self.head_act2(outputs)
        outputs = self.head_fc3(outputs)
        outputs = self.head_bc3(outputs)
        outputs = self.head_act3(outputs)
        outputs = self.head_fc4(outputs)
        outputs = self.head_bc4(outputs)
        outputs = self.head_act4(outputs)
        outputs = self.head_fc5(outputs)
        outputs = self.head_bc5(outputs)
        outputs = self.head_act5(outputs)
        outputs = self.head_fc6(outputs)
        outputs = self.head_bc6(outputs)
        outputs = self.head_act6(outputs)
        outputs = self.head_fc7(outputs)
        
        confidences_logits, logits_traj, logits_vel = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj : (self.n_traj * 2 * self.time_limit + self.n_traj)],
            outputs[:, (self.n_traj * 2 * self.time_limit + self.n_traj) : ]
        )
        # print("shape outputs: ", outputs.shape)        
        # print("shape logits_traj: ", logits_traj.shape)
        # print("shape logits_vel: ", logits_vel.shape)

        logits_traj = logits_traj.view(-1, self.n_traj, self.time_limit, 2)
        logits_vel = logits_vel.view(-1, self.n_traj, self.time_limit, 2)
        return confidences_logits, logits_traj, logits_vel


def pytorch_neg_multi_log_likelihood_batch(gt_traj, logits_traj, gt_xy_vel, logits_vel, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt_traj = torch.unsqueeze(gt_traj, 1)  # add modes
    gt_vel = torch.unsqueeze(gt_xy_vel, 1)  # add modes
    
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error_traj = torch.sum(
        ((gt_traj - logits_traj) * avails) ** 2, dim=-1
    )  # reduce coords and use availability
    error_vel = torch.sum(
        ((gt_vel - logits_vel) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error_traj = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error_traj, dim=-1
        )  # reduce time

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error_vel = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error_vel, dim=-1
        )  # reduce time
    
    # error (batch_size, num_modes)
    error_traj = -torch.logsumexp(error_traj, dim=-1, keepdim=True)
    error_vel = -torch.logsumexp(error_vel, dim=-1, keepdim=True)
    # print("error_traj: ",error_traj)
    # print("error_vel: ",error_vel)
    error = error_traj + error_vel

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
        xy_vel = data["vel_xy"]
        xy_vel = xy_vel.flatten()
        gt_xy_vel = data["gt_vel_xy"]

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, xy_vel, trajectory, gt_xy_vel, is_available, data["vector_data"]

        return raster, xy_vel, trajectory, gt_xy_vel,  is_available


def main():
    args = parse_args()

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
    model = Model(
        model_name, in_channels=args.in_channels, time_limit=time_limit, n_traj=n_traj
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

    for iteration in progress_bar:
        model.train()
        try:
            x_traj, x_xy_vel, y_traj,y_xy_vel, is_available = next(tr_it)
        except StopIteration:
            tr_it = iter(dataloader)
            x_traj, x_xy_vel, y_traj,y_xy_vel, is_available = next(tr_it)

        x_traj, x_xy_vel, y_traj,y_xy_vel, is_available = map(lambda x: x.cuda(), (x_traj, x_xy_vel, y_traj,y_xy_vel, is_available))

        optimizer.zero_grad()

        confidences_logits, logits_traj, logits_vel = model(x_traj,x_xy_vel)

        loss = pytorch_neg_multi_log_likelihood_batch(
            y_traj, logits_traj,y_xy_vel, logits_vel, confidences_logits, is_available
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
                for x_traj, x_xy_vel, y_traj, y_xy_vel, is_available in val_dataloader:
                    x_traj, x_xy_vel, y_traj,y_xy_vel, is_available = map(lambda x: x.cuda(), (x_traj, x_xy_vel, y_traj,y_xy_vel, is_available))

                    confidences_logits, logits_traj, logits_vel = model(x_traj,x_xy_vel)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        y_traj, logits_traj,y_xy_vel, logits_vel, confidences_logits, is_available
                    )
                    val_losses.append(loss.item())

                summary_writer.add_scalar("dev/loss", np.mean(val_losses), iteration)

            # saver("model_last.pth")

            mean_val_loss = np.mean(val_losses)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                saver("model_best.pth")

                model.eval()
                with torch.no_grad():
                    traced_model = torch.jit.trace(
                        model,(
                         torch.rand(1, args.in_channels, args.img_res, args.img_res).cuda(),
                         torch.rand(1,22).cuda()
                        ),
                    )

                traced_model.save(os.path.join(path_to_save, "model_best.pt"))
                del traced_model


if __name__ == "__main__":
    main()
