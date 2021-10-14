import qecstruct as qs
import qeclab as ql
import numpy as np

# First, we build a decoder for the repetition code
class Decoder:
    # This is the important function. It should always have
    # that signature. Also, it is expected that both 'message' and
    # the returned value are qecstruct.BinaryVector.
    def decode(self, message):
        if message.weight() <= len(message) / 2:
            return qs.BinaryVector.zeros(len(message))
        else:
            return qs.BinaryVector.ones(len(message))
            
# We build the experiment for a given code length and error probability.
# One experiment is one data point in a threshold plot.
def build_experiment(length, probability):
    code = qs.repetition_code(length)
    decoder = Decoder()
    noise = qs.BinarySymmetricChannel(probability)
    return ql.LinearDecodingExperiment(code, decoder, noise)

# We build a laboratory containing multiple experiments.
# Optionally, we can provide a seed for the rng and 
# the number of processes (CPUs) to use.
labo = ql.Laboratory(rng_seed=42, num_processes=1)
for length in [3, 5, 7, 9]:
    for probability in np.linspace(0.1, 1.0, 10):
        labo.add_experiment(build_experiment(length, probability))


# We run all experiments while the number of samples per experiment
# is smaller than 100.
results = labo.run_all_while(lambda stat: stat.num_samples < 100)

