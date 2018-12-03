To run the program, enter:

python main.py

This will begin generating reviews for the best GRU model at temperatures [0.4, 0.01, 10]. 
These generated files will be outputted to "reviews_tau_0.4.txt", "reviews_tau_0.01.txt", 
and "reviews_tau_10.txt", respectively. If you wish to train a model, then you will need 
to go into "main.py" and uncomment the corresponding training code block. Then run

python main.py

again. After each model is done training, it will save a ".pt" file to the "/models" 
folder. For your use, all training models are already stored there. 

As an aside, any output graphs are saved into the "/images" folder.
