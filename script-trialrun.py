if __name__ == "__main__":    
    
    print('*** script start')

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
        selfplayprocesses=6,
        searchthreadcount=25, 
        searchcount=500,
        virtualloss=0.2,
        gamelog_dump_threshold = 10)   
    TRAIN_CONFIG = gymdata.TrainConfig(
        epochs = 500, 
        batchsize = 100, 
        min_samplecount = 400,
        max_samplecount = 1000, 
        max_sampleage = 1, # sett to '-1' here to  disable training
        validation_split = 0.1,
        use_gpu = True, 
        gpu_memorylimit = None)
   
    GYM_SETTINGS = gymdata.GymConfig(PREDICT_CONFIG, SELFPLAY_CONFIG, TRAIN_CONFIG)
    GYM = gamegym.GameGym(
        session_path = '.session', 
        graph = SG, 
        # remove the intialmodelpath if continuing from an exisiting session - else the model will be overwritten
        intialmodelpath=f"D:\santorini\initialmodels\santorini\env(3-1)\model_50x5_ADAM_LR=1E-5_DR=0.2_L2=0.1", 
        gym_config = GYM_SETTINGS)
    
    # run
    GYM.resume()
    print('*** script end')
