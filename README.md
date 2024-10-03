![image](https://github.com/user-attachments/assets/2938703b-0125-4657-9808-319b512624f8)

This is a basic implementation of a nanometer- and microsecond- scale simulation model of glycolysis using CUDA. It has a visualization utility built in to visualize the molecular motion in 3D space.

#Current timing benchmarks for RTX 3070#
Frame Timings:
  Total Loop Time: 15.529 ms
  Render Time: 2.313 ms
  Simulation Time: 13.216 ms
Simulation Step Timings:
  Memory Allocation: 0.013 ms
  Memory Copy to Device: 0.579 ms
  Assign Molecules to Cells: 0.008 ms
  Apply Forces and Update Positions: 0.008 ms
  Handle Interactions: 11.970 ms
  Memory Copy from Device: 0.058 ms
  Process Creation/Deletion Flags: 0.024 ms
