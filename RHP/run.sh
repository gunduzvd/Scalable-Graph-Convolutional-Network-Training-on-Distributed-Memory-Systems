
list=(1000 2000 4000 8000)

for((i=4;i<6;i++))
do
k=$((2**($i+4)))
echo "$k ${list[i]}"
#python main.py -p data/football/football.mtx -k $k -s  ${list[i]}  -b 64 -h  ${list[i]} >> results/result.$k
python main.py -p data/com-Amazon/com-Amazon.mtx -k $k -s 10000 -b 20000 -h 10000  >> results/result.amazon.$k
done
