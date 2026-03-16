#!/bin/bash
set -e

# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 nohup python -u run_my.py --do_align --alignment dpo --align_attack NaiveCompletion -m /workspace/huggingface_model/llama-7b_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-25-23-50-41 > ours_secdpo_llama_7b_role.log 2>&1 &
# pid1=$!

# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup python -u run_my.py --do_align --alignment dpo --align_attack NaiveCompletion -m /workspace/huggingface_model/Meta-Llama-3-8B_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-26-07-35-32 > ours_secdpo_meta_8b_role.log 2>&1 &
# pid1=$!

# wait $pid1   # 等第一个进程结束
# sleep 120

# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 nohup python -u run_my.py --do_align --alignment dpo --align_attack NaiveCompletion -m /workspace/huggingface_model/Mistral-7B-v0.1_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-26-03-37-17 > ours_secdpo_mis_7b_role.log 2>&1 &

# nohup python -u run_my.py --do_sft --sft_attack RoleSpclSpclSpcl_NaiveIgnoreCompletion -m /workspace/huggingface_model/Meta-Llama-3-8B > ours_sft_llama3_8b_train_N_I_C.log 2>&1 &
# pid3=$!

# wait $pid3
# sleep 120

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup python -u run_my.py --do_sft --sft_attack RoleSpclSpclSpcl_NaiveCompletion -m /workspace/huggingface_model/Mistral-7B-v0.1 > ours_sft_mis7b_train_NIC_new_5.log 2>&1 &
pid1=$!

wait $pid1
sleep 120

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup python -u run_my.py --do_sft --sft_attack RoleSpclSpclSpcl_NaiveCompletion -m /workspace/huggingface_model/Meta-Llama-3-8B > ours_sft_meta3_8b_train_NIC_new_5.log 2>&1 &
pid2=$!

wait $pid2
sleep 120

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup python -u run_my.py --do_sft --sft_attack RoleSpclSpclSpcl_NaiveCompletion -m /workspace/huggingface_model/llama-7b > ours_sft_llama7b_train_NIC_new_5.log 2>&1 &
