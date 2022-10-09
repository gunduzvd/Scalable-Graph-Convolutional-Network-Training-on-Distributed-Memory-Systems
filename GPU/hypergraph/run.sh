MAIN_DATA_DIR=/data/gunduz/dgl-output/data

for DATA_DIR in `ls ${MAIN_DATA_DIR}`
do
 echo "${MAIN_DATA_DIR}/${DATA_DIR}"
 FILE_A="${DATA_DIR}.A.mtx"
 echo ${FILE_A}
 for k in 15 21 3 9 27 2
 do
  echo "./gcnhgp -a ${MAIN_DATA_DIR}/${DATA_DIR}/${FILE_A} -o ./parts -k ${k}"
  ./gcnhgp -a ${MAIN_DATA_DIR}/${DATA_DIR}/${FILE_A} -o ./parts -k ${k}
 done
done
