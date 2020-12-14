python gen_dist.py

for k in 1 2 3
do
	for i in {0..31}
	do
        cp dist.log dist_emb_${i}.log
	done
done
