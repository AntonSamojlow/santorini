# standard modules
import logging
import os
import datetime
import random

if __name__ == "__main__":    
    # root logging setup
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    fh = logging.FileHandler('{}.{}.log'.format(__file__, datetime.datetime.now().strftime("%d-%m-%Y %H_%M_%S")))
    fh.setLevel(logging.INFO)
    logging.basicConfig(
        handlers=[ch, fh],
        level=logging.DEBUG, 
        format='%(asctime)s {%(processName)s} [%(name)s] %(levelname)s: %(message)s')
    logging.info('*** script start')

    # custom modules
    import santorini
    import gamegym
    import gymdata

    # the game graph we operate on
    SG = santorini.SanGraph(env=santorini.Environment(3,1))
    
    # configuration
    SELFPLAY_CONFIG = gymdata.SelfPlayConfig(
        searchthreadcount=25, 
        searchcount=100, 
        virtualloss=0.2)
    TRAIN_CONFIG = gymdata.TrainConfig(
        epochs = 100, 
        batchsize = 100, 
        maxsamplecount = 1000, 
        maxsampleage = 2,
        use_gpu = True, 
        gpu_memorylimit = 2048)
    PREDICT_CONFIG = gymdata.PredictConfig(
        batchsize=5,
        use_gpu = True, 
        gpu_memorylimit = 2048)
    GYM_SETTINGS = gymdata.GymConfig(PREDICT_CONFIG, SELFPLAY_CONFIG, TRAIN_CONFIG)
    GYM = gamegym.GameGym(
        session_path = '.session', 
        graph = SG, 
        intialmodelpath ='initialmodels/dim{}unitspp{}layers50x50'.format(SG.env.dimension, SG.env.units_per_player),
        gym_config = GYM_SETTINGS)
    
    # run
    GYM.resume()

    logging.info('*** script end')
