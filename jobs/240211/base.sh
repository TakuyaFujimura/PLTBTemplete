############SETTINGS############
name="plain"
version=$1
exp_yaml=$1
################################
project_dir="/data/group1/${USER}/project_tmp"
repo="PLTBTemplete"
num_workers=0
################################

echo "num_workers was set to ${num_workers}"
echo "project_dir was set to ${project_dir}"
source "${project_dir}/${repo}/venv/bin/activate"
# source "/data/group1/z44722r/project02/venv/bin/activate"
exp_root="${project_dir}/${repo}/results"
echo ${exp_root}
cd "${project_dir}/${repo}"

echo "################################################"

python train.py experiments="${exp_yaml}" \
'path.exp_root='${exp_root}'' 'name='${name}'' \
'num_workers='${num_workers}'' 'version='${version}'' \
'refresh_rate=100'
