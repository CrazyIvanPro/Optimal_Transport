for dataset in 'random' 'caffarelli' 'ellipse'
do
    for n in 64 128 256 512 1024 2048 # 4096
    do
        for method in 'primal' 'dual' 'interior'
        do
            python test_raw_gurobi.py --n $n --dataset $dataset --method $method
        done
    done
done

for dataset in 'DOTmark'
do
    for n in 64 128 256 512 1024 2048 # 4096
    do
        for imageclass in `ls ./DOT`
        do
            for method in 'primal' 'dual' 'interior'
            do
                python test_raw_gurobi.py --n $n --dataset $dataset --imageclass $imageclass --method $method 
            done
        done
    done
done