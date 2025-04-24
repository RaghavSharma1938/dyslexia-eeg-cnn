import pytorch_lightning as pl, torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.data.dataset import DyslexiaDataModule
from src.models.cnn import SimpleCNN
from sklearn.metrics import accuracy_score

class Lit(pl.LightningModule):
    def __init__(self, n_ch):
        super().__init__()
        self.model = SimpleCNN(n_ch)
        self.save_hyperparameters()

    def forward(self,x): return self.model(x)

    def _step(self,batch,stage):
        x,y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits,y)
        acc  = accuracy_score(y.cpu(), logits.argmax(1).cpu())
        self.log_dict({f"{stage}_loss":loss, f"{stage}_acc":acc},
                      prog_bar=(stage=="val"), on_epoch=True)
        return loss

    def training_step(self,batch,_):   return self._step(batch,"train")
    def validation_step(self,batch,_): self._step(batch,"val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                         patience=4, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler":{"scheduler":sch,"monitor":"val_acc"}}

def main():
    dm = DyslexiaDataModule(batch=128)   # num_workers fixed at 0 internally
    dm.setup()
    model = Lit(dm.n_channels)

    cb_es = EarlyStopping(monitor="val_acc", mode="max", patience=10)
    cb_ck = ModelCheckpoint(dirpath="checkpoints",
                            monitor="val_acc", mode="max",
                            filename="epoch{epoch:02d}-val{val_acc:.3f}")

    trainer = pl.Trainer(max_epochs=60, callbacks=[cb_es, cb_ck],
                         log_every_n_steps=10)
    trainer.fit(model, dm)

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()
