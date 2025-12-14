# Training script for IRanker (Iterative Deletion Ranker)
# Uses data from data/iterative_ranking/
# Set DATA_DIR to data/iterative_ranking before running

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=gae \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=8 \
data.val_batch_size=8 \
data.max_prompt_length=1400 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size=8 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.grad_offload=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.rollout.temperature=1 \
actor_rollout_ref.actor.use_kl_loss=True \
critic.optim.lr=2e-6 \
critic.model.path=$BASE_MODEL \
critic.ppo_mini_batch_size=32 \
critic.ppo_micro_batch_size=8 \
critic.model.fsdp_config.param_offload=True \
critic.model.fsdp_config.grad_offload=True \
critic.model.fsdp_config.optimizer_offload=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.val_before_train=True \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=100 \
trainer.project_name=Ranking-FM \
trainer.experiment_name=IRanker-${EXPERIMENT_NAME} \
trainer.total_training_steps=1501 \
trainer.total_epochs=3 2>&1 | tee verl_demo.log