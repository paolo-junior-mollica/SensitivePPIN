from drug_disease_validation.src.dgn_model.dgn.dgn import DGN
from drug_disease_validation.src.dgn_model.datamodule import GraphDataModule
from lightning import Trainer
import torch
from drug_disease_validation.src.dgn_model.runtime import get_runtime_config, get_trainer_kwargs

def load_and_predict(config, ckpt, datalist):

    if 'aggr' not in config.keys():
        config['aggr'] = config['SAGE_aggr']
    if 'uniform_bound' not in config.keys():
        config['uniform_bound'] = None
    if 'weight_initializer' not in config.keys():
        config['weight_initializer'] = 'kaiming_uniform'
    
    config['weighted_sampler'] = None
    
    input_dim = datalist[0].x.shape[1]
    output_dim = 1
    data = GraphDataModule(datalist, datalist, datalist, config)
    data.setup()
    runtime = get_runtime_config(torch)

    model = DGN.load_from_checkpoint(
        ckpt,
        input_dim=input_dim,
        output_dim=output_dim,
        config=config,
        map_location=torch.device(runtime.device_type),
        strict=False,
    )

    trainer = Trainer(enable_progress_bar=False, logger=False, **get_trainer_kwargs(runtime))
    print("Model type: ", type(model.eval()))
    
    predictions = trainer.predict(model, data.test_dataloader())
    predictions = torch.cat(predictions)
    
    return predictions
