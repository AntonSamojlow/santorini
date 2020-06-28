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
        batchsize=50,
        use_gpu = False, 
        gpu_memorylimit = None)
    SELFPLAY_CONFIG = gymdata.SelfPlayConfig(
        selfplayprocesses=4,
        searchthreadcount=100, 
        searchcount=500,
        virtualloss=0.1,
        record_dumpbatchsize = 250)   
    TRAIN_CONFIG = gymdata.TrainConfig(
        epochs = 500, 
        batchsize = 100, 
        min_samplecount = 5000,
        max_samplecount = 250000, 
        max_sampleage = -1, # this disables training
        validation_split = 0.2,
        use_gpu = True, 
        gpu_memorylimit = None)
   
    GYM_SETTINGS = gymdata.GymConfig(PREDICT_CONFIG, SELFPLAY_CONFIG, TRAIN_CONFIG)
    GYM = gamegym.GameGym(
        session_path = '.session', 
        graph = SG, 
        intialmodelpath ="initialmodels/dim{}_upp{}_init".format(SG.env.dimension, SG.env.units_per_player),
        gym_config = GYM_SETTINGS)
    
    # run
    GYM.resume()

    logging.info('*** script end')
