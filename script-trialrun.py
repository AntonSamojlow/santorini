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
    print('*** script end')
