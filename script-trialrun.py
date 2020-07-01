# standard modules
import logging
import os
import datetime

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

    # parameters
    SG = santorini.SanGraph(santorini.Environment(dimension=3, units_per_player=1))
        
    # configuration
    PREDICT_CONFIG = gymdata.PredictConfig(
        batchsize=75,
        use_gpu = False, 
        gpu_memorylimit = None)
    SELFPLAY_CONFIG = gymdata.SelfPlayConfig(
        selfplayprocesses=8,
        searchthreadcount=25, 
        searchcount=100,
        virtualloss=0.2,
        gamelog_dump_threshold = 10)   
    TRAIN_CONFIG = gymdata.TrainConfig(
        epochs = 500, 
        batchsize = 100, 
        min_samplecount = 500,
        max_samplecount = 1000, 
        max_sampleage = 2, # -1 here disables training
        validation_split = 0.1,
        use_gpu = True, 
        gpu_memorylimit = None)
   
    GYM_SETTINGS = gymdata.GymConfig(PREDICT_CONFIG, SELFPLAY_CONFIG, TRAIN_CONFIG)
    GYM = gamegym.GameGym(
        session_path = '.session', 
        graph = SG, 
        intialmodelpath=f"initialmodels/dim{SG.env.dimension}_upp{SG.env.units_per_player}_50x5_ADAM", 
        gym_config = GYM_SETTINGS)
    
    # run
    GYM.resume()
    logging.info('*** script end')
