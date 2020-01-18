for dataset in 'random' 'caffarelli' 'ellipse'
do
    for n in 64 128 256 512 1024 2048 # 4096
    do
        python test_blockca.py --n $n --dataset $dataset
    done
done

for dataset in 'DOTmark'
do
    for n in 64 128 256 512 1024 2048 # 4096
    do
        for imageclass in `ls ./DOT`
        do
            python test_blockca.py --n $n --dataset $dataset --imageclass $imageclass
        done
    done
done