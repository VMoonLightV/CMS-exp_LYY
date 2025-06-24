# CMS-exp_LYY

My work on CMS experiment is searching Higgs decaying to a pair of muons event. I used to work on bamboo framework and HiggsDNA framework, the job is similar, I just transition from bamboo to HiggsDNA.

XXX_hto2mu.py is the main code I work on, I use it and some config files to processing experiment data and Monte Carlo with certain selection, then obtain the events I'm interested in.

Later I use Neural network (DNN before and transformer now) to train signal discriminator with MC events to optimize its performance, then apply it to experiment data, using statistical methods to analyse signal significance.
