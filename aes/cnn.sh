#!/bin/bash
#
#SBATCH --job-name=myPythonJobGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=hi_CNN
#SBATCH --mail-type=END
#SBATCH --mail-user=yz3464@nyu.edu
#SBATCH --output=data_cnn_lr.out
#SBATCH --gres=gpu:1

module purge

module load python/intel/2.7.12
module load keras/2.0.2
module load theano/0.9.0
module load scipy/intel/0.19.1
module load scikit-learn/intel/0.18.1
module load numpy/intel/1.13.1
module load nltk/3.2.2


checkdir=../checkpoint

embed_dim=50
prompt_id=1
embed_type=glove

echo $embed_type

trainfile=../data/train.tsv
devfile=../data/dev.tsv
testfile=../data/test.tsv


if [ ! -d $checkdir/preds ]; then
            mkdir -p $checkdir/preds
fi

embeddingfile=../data/glove.6B.50d.txt

nb_epochs=12

echo "Using embedding ${embeddingfile}"

 # THEANO_FLAGS='floatX=float32,device=cpu'
python hi_CNN.py --fine_tune --embedding $embed_type --embedding_dict $embeddingfile --embedding_dim ${embed_dim} \
              --num_epochs $nb_epochs --batch_size 32 --nbfilters 100 --filter1_len 5 --filter2_len 3 \
                      --optimizer rmsprop --learning_rate 0.0001 --dropout 0.5 \
                              --oov embedding --l2_value 0.01 --checkpoint_path $checkdir \
                                      --train $trainfile --dev $devfile --test $testfile --prompt_id $prompt_id --train_flag


