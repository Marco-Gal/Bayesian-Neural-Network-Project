import slang_mnist_complete as slang_mnist_complete
experiment_base = slang_mnist_complete.slang_complete

from slang_mnist_experiment import experiment_name, variants

from mnist.mnist_helpers import get_accuracy
import pandas as pd

#############################
## Compute test accuracies ##
#############################

accuracies = []

for i, variant in enumerate(variants):
    accuracies.append(get_accuracy(experiment_base, experiment_name, variant, mc_10_multiplier = 100, data_set='mnist'))
    print("Index [{}/{}] Done!".format(1+i, len(variants)))

df_final = pd.DataFrame(dict(variant=variants, accuracy=accuracies))
df_final['error'] = ['{:.2f}%'.format(err) for err in (1 - df_final['accuracy'].values) * 100]
df_final.to_csv("table_3.csv", index=False)