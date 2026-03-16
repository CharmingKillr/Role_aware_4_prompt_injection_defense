#!/bin/bash
set -e

# ASCEND_RT_VISIBLE_DEVICES=4 python -u run_my.py --do_test --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/huggingface_model/llama-7b_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-25-23-50-41_dpo_NaiveCompletion_2025-12-27-02-47-23  > test_ours_sec_llama7b_NIC.log 2>&1 &
# pid1=$!

# ASCEND_RT_VISIBLE_DEVICES=5 python -u run_my.py --do_test --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/huggingface_model/Meta-Llama-3-8B_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-26-07-35-32_dpo_NaiveCompletion_2025-12-27-02-47-23  > test_ours_sec_meta8b_NIC.log 2>&1 &
# pid2=$!

# wait $pid1   # 等第一个进程结束
# sleep 120

# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 nohup python -u run_my.py --do_align --alignment dpo --align_attack NaiveCompletion -m /workspace/huggingface_model/Mistral-7B-v0.1_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-26-03-37-17 > ours_secdpo_mis_7b_new.log 2>&1 &

# wait $pid2   # 等第二个进程结束
# sleep 60

# ASCEND_RT_VISIBLE_DEVICES=7 python -u run_my.py --do_test --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/huggingface_model/Meta-Llama-3-8B_RoleSpclSpclSpcl_NaiveIgnoreCompletion_2025-12-26-07-35-32  > test_ours_meta-8b_NIC.log 2>&1 &


# nohup python -u run_my.py --do_sft --sft_attack RoleSpclSpclSpcl_NaiveIgnoreCompletion -m /workspace/huggingface_model/Meta-Llama-3-8B > ours_sft_llama3_8b_train_N_I_C.log 2>&1 &
# pid3=$!

# wait $pid3
# sleep 120

#prompt-based-defense:  'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'

ASCEND_RT_VISIBLE_DEVICES=3 python -u run_my.py --do_test --defense sandwich --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14 > test_undef_meta8b_sandwich.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=4 python -u run_my.py --do_test --defense instructional --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14 > test_undef_meta8b_instructional.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=5 python -u run_my.py --do_test --defense reminder --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14 > test_undef_meta8b_reminder.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=6 python -u run_my.py --do_test --defense isolation --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14 > test_undef_meta8b_isolation.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=7 python -u run_my.py --do_test --defense incontext --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14 > test_undef_meta8b_incontext.log 2>&1 &


ASCEND_RT_VISIBLE_DEVICES=3 python -u run_my.py --do_test --defense sandwich --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08 > test_undef_mis7b_sandwich.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=4 python -u run_my.py --do_test --defense instructional --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08 > test_undef_mis7b_instructional.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=5 python -u run_my.py --do_test --defense reminder --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08 > test_undef_mis7b_reminder.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=6 python -u run_my.py --do_test --defense isolation --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08 > test_undef_mis7b_isolation.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=7 python -u run_my.py --do_test --defense incontext --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08 > test_undef_mis7b_incontext.log 2>&1 &


ASCEND_RT_VISIBLE_DEVICES=1 python -u run_my.py --do_test --defense sandwich --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20 > test_undef_llama7b_sandwich.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=1 python -u run_my.py --do_test --defense instructional --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20 > test_undef_llama7b_instructional.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=2 python -u run_my.py --do_test --defense reminder --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20 > test_undef_llama7b_reminder.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=2 python -u run_my.py --do_test --defense isolation --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20 > test_undef_llama7b_isolation.log 2>&1 &

ASCEND_RT_VISIBLE_DEVICES=0 python -u run_my.py --do_test --defense incontext --test_attack naive ignore escape_deletion escape_separation completion_other completion_othercmb completion_real completion_realcmb completion_real_base64 completion_real_chinese completion_real_spanish hackaprompt -m /workspace/SecAlign/huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20 > test_undef_llama7b_incontext.log 2>&1 &

