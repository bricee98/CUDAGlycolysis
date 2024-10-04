# CUDA Glycolysis
It's glycolysis in a box! This program simulates Brownian motion of the molecules and enzymes involved in the glycolysis pathway to show how the process progresses over time under various conditions.

It leverages parallel processing with CUDA to efficiently compute motino and pairwise interactions between many molecules - up to 100,000,000 molecules have been successfully simulated on an RTX3070.

There's also a real-time visualization built in.

## Simulation Controls
### R: Toggle visualization rendering
### Spacebar: Toggle simulation
### W: Write current molecule quantities and time to file
### L: Toggle console logs

## Visualization Controls
### W/A/S/D: Pan camera
### Q/Z: Move up/down

![image](https://github.com/user-attachments/assets/2938703b-0125-4657-9808-319b512624f8)


## Current timing benchmarks for RTX 3070
### Simulation Step Timings - 10,000 molecules:
  - Memory Allocation: 0.088 ms
  - Reset Cells: 0.064 ms
  - Memory Copy to Device: 0.087 ms
  - Assign Molecules to Cells: 0.023 ms
  - Apply Forces and Update Positions: 0.034 ms
  - Handle Bindings: 0.106 ms
  - Memory Copy from Device: 0.109 ms
  - Process Creation/Deletion Flags: 0.028 ms
  - Reset Creation/Deletion Buffers: 0.012 ms
  - Memory Copy to Device: 0.069 ms
  - Handle Reactions and Dissociations: 0.094 ms
  - Memory Copy from Device: 0.119 ms
  - Process Creation/Deletion Flags: 0.028 ms
  - Total Calculated Time: 0.862 ms

### Simulation Step Timings - 100,000,000 molecules:
  - Memory Allocation: 1.319 ms
  - Reset Cells: 1.299 ms
  - Memory Copy to Device: 366.517 ms
  - Assign Molecules to Cells: 0.026 ms
  - Apply Forces and Update Positions: 0.025 ms
  - Handle Bindings: 0.009 ms
  - Memory Copy from Device: 69.871 ms
  - Process Creation/Deletion Flags: 0.033 ms
  - Reset Creation/Deletion Buffers: 0.008 ms
  - Memory Copy to Device: 69.114 ms
  - Handle Reactions and Dissociations: 69.214 ms
  - Memory Copy from Device: 70.441 ms
  - Process Creation/Deletion Flags: 0.026 ms
  - Total Calculated Time: 647.902 ms

## Planned improvements:
- Compute molecule creation/deletion on GPU so that memory doesn't need to be copied every cycle
- Implement effects of regulatory molecules
- Implement realistic reaction/dissociation probabilities for each enzyme-substrate pair
  
![image](https://github.com/user-attachments/assets/49a140fe-c049-4b0c-bb02-7a15592f8444)
