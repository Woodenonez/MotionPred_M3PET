# Create (Jun. 30 2022) [MotionPred_M3PET]
M3PET - Multimodal Motion Prediction Energy-based Network

# 20220711
"net.py": UNet, YNet, ENet

# 20220806
Modify dataset and data handler to the new verson (traj pred only).
"dataset.py": Delete T_channel and dyn_env. Make it specific for traj prediction.

# 20220830
Compare BCE/NLL and so on. Try to finalized the workflow. Try to modify the network architecture.
Test PELU layer. NLL may have a problem that it doesn't penalize on non-target cells/pixels enough.