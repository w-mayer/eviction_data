import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class TaskType(Enum):
    CPU_BOUND = "CPU-bound"
    MEMORY_BOUND = "Memory-bound"
    IO_BOUND = "IO-bound"
    BALANCED = "Balanced"

class ResourceCalculator:
    """
    A calculator for estimating task completion times and optimal resource allocation
    """
    
    def __init__(self):
        # Define standard HPC configurations
        self.hpc_configs = {
            'small': {'cores_per_node': 24, 'memory_per_node': 128, 'cost_per_node_hour': 1.0},
            'medium': {'cores_per_node': 48, 'memory_per_node': 256, 'cost_per_node_hour': 2.0},
            'large': {'cores_per_node': 96, 'memory_per_node': 512, 'cost_per_node_hour': 4.0},
            'gpu': {'cores_per_node': 24, 'memory_per_node': 256, 'gpus_per_node': 4, 'cost_per_node_hour': 6.0}
        }
        
        # Amdahl's Law coefficient - higher means better parallelization
        self.parallelization_efficiency = 0.8
        
        # Resource bottleneck impact factors
        self.bottleneck_factors = {
            TaskType.CPU_BOUND: {'cpu': 0.8, 'memory': 0.1, 'io': 0.1},
            TaskType.MEMORY_BOUND: {'cpu': 0.2, 'memory': 0.7, 'io': 0.1},
            TaskType.IO_BOUND: {'cpu': 0.2, 'memory': 0.1, 'io': 0.7},
            TaskType.BALANCED: {'cpu': 0.4, 'memory': 0.3, 'io': 0.3}
        }
    
    def calculate_completion_time(self, task_spec, available_resources):
        """
        Calculate estimated completion time based on task specifications and available resources
        
        Parameters:
        - task_spec: dict containing task requirements
            - base_runtime_hours: baseline runtime with ideal resources
            - required_memory_gb: memory required by the task
            - required_cores: ideal number of CPU cores
            - task_type: TaskType enum indicating bottleneck characteristics
            - parallelizable_percentage: percentage of task that can be parallelized (0-100)
        
        - available_resources: dict containing available resources
            - memory_gb: available memory in GB
            - cpu_cores: available CPU cores
            - io_speed_mbps: I/O throughput in MB/s
        
        Returns:
        - dict with estimated completion time and limiting factors
        """
        # Extract task requirements
        base_runtime = task_spec['base_runtime_hours']
        required_memory = task_spec['required_memory_gb']
        required_cores = task_spec['required_cores']
        task_type = task_spec['task_type']
        parallelizable_pct = task_spec['parallelizable_percentage'] / 100.0
        
        # Calculate resource constraint factors
        memory_factor = self._calculate_memory_constraint(required_memory, available_resources['memory_gb'])
        cpu_factor = self._calculate_cpu_constraint(required_cores, available_resources['cpu_cores'], parallelizable_pct)
        io_factor = self._calculate_io_constraint(task_spec.get('io_intensity', 0.5), available_resources['io_speed_mbps'])
        
        # Get task-specific weightings for different resource types
        weights = self.bottleneck_factors[task_type]
        
        # Calculate weighted constraint factor
        constraint_factor = (
            weights['cpu'] * cpu_factor +
            weights['memory'] * memory_factor +
            weights['io'] * io_factor
        )
        
        # Estimated completion time (never less than base runtime)
        estimated_time = max(base_runtime, base_runtime * constraint_factor)
        
        # Determine limiting factor
        factors = {
            'CPU': cpu_factor * weights['cpu'],
            'Memory': memory_factor * weights['memory'],
            'I/O': io_factor * weights['io']
        }
        limiting_factor = max(factors, key=factors.get)
        
        return {
            'estimated_hours': estimated_time,
            'limiting_factor': limiting_factor,
            'constraint_factors': {
                'memory': memory_factor,
                'cpu': cpu_factor,
                'io': io_factor
            },
            'weighted_constraint': constraint_factor
        }
    
    def _calculate_memory_constraint(self, required_gb, available_gb):
        """Calculate memory constraint factor"""
        if required_gb <= available_gb:
            return 1.0
        else:
            # Memory constraints cause swapping/thrashing, which has non-linear performance impact
            # Simplified model: quadratic penalty when exceeding available memory
            memory_ratio = required_gb / available_gb
            return memory_ratio * memory_ratio
    
    def _calculate_cpu_constraint(self, required_cores, available_cores, parallelizable_pct):
        """Calculate CPU constraint factor using Amdahl's Law"""
        if required_cores <= available_cores:
            return 1.0
        
        # Non-parallelizable portion
        serial_fraction = 1.0 - parallelizable_pct
        
        # Amdahl's Law
        speedup = 1 / (serial_fraction + (parallelizable_pct / (available_cores / required_cores)))
        
        # Constraint factor (inverse of speedup relative to ideal case)
        constraint = required_cores / (available_cores * speedup)
        
        return max(1.0, constraint)
    
    def _calculate_io_constraint(self, io_intensity, available_io_speed):
        """Calculate I/O constraint factor"""
        # Simple linear model for I/O constraints
        # Assumes baseline of 500 MB/s
        baseline_io = 500
        
        if io_intensity <= 0.2:  # Low I/O intensity
            return 1.0
        else:
            io_ratio = available_io_speed / baseline_io
            # Adjust based on I/O intensity
            constraint = 1 + (io_intensity * (1 - io_ratio))
            return max(1.0, constraint)
    
    def should_request_hpc(self, task_spec, available_resources, deadline_hours=None):
        """
        Determine if HPC resources should be requested
        
        Parameters:
        - task_spec: task specification dictionary
        - available_resources: available resource dictionary
        - deadline_hours: optional deadline in hours
        
        Returns:
        - dict with decision and reasoning
        """
        # Calculate estimated completion time with current resources
        time_estimate = self.calculate_completion_time(task_spec, available_resources)
        
        # Check if current resources are sufficient
        memory_sufficient = task_spec['required_memory_gb'] <= available_resources['memory_gb']
        
        # Determine if task is HPC-suitable
        hpc_suitable = (
            task_spec['parallelizable_percentage'] > 40 and  # At least 40% parallelizable
            task_spec['base_runtime_hours'] > 1.0  # At least 1 hour baseline runtime
        )
        
        # Decision logic with deadline if provided
        if deadline_hours:
            can_meet_deadline = time_estimate['estimated_hours'] < deadline_hours
            
            if not memory_sufficient:
                return {
                    'request_hpc': True,
                    'reason': 'Task exceeds local memory capacity',
                    'urgency': 'Critical'
                }
            elif not can_meet_deadline and hpc_suitable:
                return {
                    'request_hpc': True,
                    'reason': f'Cannot meet deadline of {deadline_hours} hours with local resources',
                    'urgency': 'High'
                }
            elif time_estimate['estimated_hours'] > 24 and hpc_suitable:
                return {
                    'request_hpc': True,
                    'reason': 'Task runtime exceeds 24 hours on local resources',
                    'urgency': 'Medium'
                }
            else:
                return {
                    'request_hpc': False,
                    'reason': 'Local resources sufficient for deadline',
                    'urgency': 'Low'
                }
        else:
            # Decision without deadline constraint
            if not memory_sufficient:
                return {
                    'request_hpc': True,
                    'reason': 'Task exceeds local memory capacity',
                    'urgency': 'High'
                }
            elif time_estimate['estimated_hours'] > 48 and hpc_suitable:
                return {
                    'request_hpc': True,
                    'reason': 'Task runtime exceeds 48 hours on local resources',
                    'urgency': 'Medium'
                }
            elif time_estimate['estimated_hours'] > 8 and hpc_suitable:
                return {
                    'request_hpc': True,
                    'reason': 'Task runtime significant; HPC would improve throughput',
                    'urgency': 'Low'
                }
            else:
                return {
                    'request_hpc': False,
                    'reason': 'Local resources sufficient or task not suitable for HPC',
                    'urgency': 'Low'
                }
    
    def recommend_hpc_resources(self, task_spec, deadline_hours=None):
        """
        Recommend optimal HPC resources for a task
        
        Parameters:
        - task_spec: task specification dictionary
        - deadline_hours: optional deadline in hours
        
        Returns:
        - dict with recommended HPC configuration
        """
        # Extract task requirements
        required_memory = task_spec['required_memory_gb']
        required_cores = task_spec['required_cores']
        base_runtime = task_spec['base_runtime_hours']
        parallelizable_pct = task_spec['parallelizable_percentage'] / 100.0
        task_type = task_spec['task_type']
        
        # For GPU tasks, use GPU nodes
        if task_type == 'GPU' or task_spec.get('requires_gpu', False):
            return self._recommend_gpu_resources(task_spec, deadline_hours)
        
        # Determine optimal node type
        if required_memory > 384:
            node_type = 'large'
        elif required_memory > 192:
            node_type = 'medium'
        else:
            node_type = 'small'
        
        # Get node specifications
        node_specs = self.hpc_configs[node_type]
        cores_per_node = node_specs['cores_per_node']
        memory_per_node = node_specs['memory_per_node']
        
        # Calculate nodes needed for memory
        nodes_for_memory = math.ceil(required_memory / memory_per_node)
        
        # Calculate nodes needed for CPU
        if parallelizable_pct > 0.9:  # Highly parallelizable
            # For highly parallelizable tasks, more cores help significantly
            # but with diminishing returns
            target_cores = min(required_cores * 2, required_cores + 96)
            nodes_for_cpu = math.ceil(target_cores / cores_per_node)
        else:
            # Apply Amdahl's Law for parallelizable fraction
            serial_fraction = 1.0 - parallelizable_pct
            
            # Optimal core count based on parallelizable portion
            optimal_cores = min(
                required_cores * 2,  # Don't go too far beyond required
                required_cores + (parallelizable_pct * required_cores)
            )
            
            nodes_for_cpu = math.ceil(optimal_cores / cores_per_node)
        
        # Take the maximum of nodes needed for memory and CPU
        nodes_required = max(nodes_for_memory, nodes_for_cpu)
        
        # Adjust for deadline if provided
        if deadline_hours:
            # Approximate speedup from parallelization
            serial_fraction = 1.0 - parallelizable_pct
            max_speedup = 1 / serial_fraction if serial_fraction > 0 else float('inf')
            
            # Calculate minimum nodes needed to meet deadline
            min_speedup_needed = base_runtime / deadline_hours
            
            if min_speedup_needed < max_speedup:
                # We can meet the deadline with enough nodes
                # Solve for number of nodes based on Amdahl's Law
                if serial_fraction > 0:
                    parallelizable_speedup_needed = parallelizable_pct / (min_speedup_needed - serial_fraction)
                    cores_needed = required_cores / parallelizable_speedup_needed
                    deadline_nodes = math.ceil(cores_needed / cores_per_node)
                    nodes_required = max(nodes_required, deadline_nodes)
        
        # Ensure at least one node
        nodes_required = max(1, nodes_required)
        
        # Calculate total resources
        total_cores = nodes_required * cores_per_node
        total_memory = nodes_required * memory_per_node
        
        # Estimate runtime with these resources
        estimated_runtime = self._estimate_runtime_with_resources(
            task_spec, {'cpu_cores': total_cores, 'memory_gb': total_memory}
        )
        
        # Calculate cost estimate
        cost_estimate = nodes_required * node_specs['cost_per_node_hour'] * estimated_runtime
        
        return {
            'node_type': node_type,
            'nodes_required': nodes_required,
            'total_cores': total_cores,
            'total_memory_gb': total_memory,
            'estimated_runtime_hours': estimated_runtime,
            'cost_estimate': cost_estimate,
            'slurm_parameters': self._generate_slurm_parameters(
                nodes_required, node_type, estimated_runtime
            )
        }
    
    def _recommend_gpu_resources(self, task_spec, deadline_hours=None):
        """Recommend GPU resources for GPU-intensive tasks"""
        node_specs = self.hpc_configs['gpu']
        
        # GPU tasks often need 1-4 GPUs
        gpus_required = task_spec.get('required_gpus', 1)
        
        # Calculate number of nodes based on GPUs required
        nodes_required = math.ceil(gpus_required / node_specs['gpus_per_node'])
        
        # Check if memory is sufficient
        memory_required = task_spec['required_memory_gb']
        memory_per_node = node_specs['memory_per_node']
        
        nodes_for_memory = math.ceil(memory_required / memory_per_node)
        nodes_required = max(nodes_required, nodes_for_memory)
        
        # Calculate total resources
        total_cores = nodes_required * node_specs['cores_per_node']
        total_memory = nodes_required * memory_per_node
        total_gpus = nodes_required * node_specs['gpus_per_node']
        
        # Estimate runtime
        # GPU tasks scale differently - we need to consider GPU scaling factor
        gpu_speedup = min(gpus_required, total_gpus) / gpus_required
        parallelizable_pct = task_spec['parallelizable_percentage'] / 100.0
        
        # Apply modified Amdahl's Law for GPU tasks
        serial_fraction = 1.0 - parallelizable_pct
        speedup = 1 / (serial_fraction + (parallelizable_pct / gpu_speedup))
        
        estimated_runtime = task_spec['base_runtime_hours'] / speedup
        
        # Calculate cost
        cost_estimate = nodes_required * node_specs['cost_per_node_hour'] * estimated_runtime
        
        return {
            'node_type': 'gpu',
            'nodes_required': nodes_required,
            'total_cores': total_cores,
            'total_memory_gb': total_memory,
            'total_gpus': total_gpus,
            'estimated_runtime_hours': estimated_runtime,
            'cost_estimate': cost_estimate,
            'slurm_parameters': self._generate_slurm_parameters(
                nodes_required, 'gpu', estimated_runtime, gpus_required
            )
        }
    
    def _estimate_runtime_with_resources(self, task_spec, resources):
        """Estimate runtime with specified resources"""
        # Extract task requirements
        base_runtime = task_spec['base_runtime_hours']
        required_cores = task_spec['required_cores']
        parallelizable_pct = task_spec['parallelizable_percentage'] / 100.0
        
        # Calculate speedup based on available cores
        serial_fraction = 1.0 - parallelizable_pct
        available_cores = resources['cpu_cores']
        
        # Apply Amdahl's Law
        if serial_fraction > 0:
            speedup = 1 / (serial_fraction + (parallelizable_pct * required_cores / available_cores))
        else:
            # Perfect parallelization (theoretical)
            speedup = available_cores / required_cores
        
        # Apply efficiency factor for realistic scaling
        effective_speedup = speedup * self.parallelization_efficiency
        
        # Calculate estimated runtime
        estimated_runtime = base_runtime / effective_speedup
        
        # Ensure runtime is never less than ideal case
        return max(estimated_runtime, base_runtime * serial_fraction)
    
    def _generate_slurm_parameters(self, nodes, node_type, runtime_hours, gpus=None):
        """Generate SLURM batch script parameters"""
        # Round up runtime to nearest hour and add 20% buffer
        runtime_with_buffer = math.ceil(runtime_hours * 1.2)
        
        # Format as HH:MM:00
        hours = runtime_with_buffer
        minutes = 0
        time_str = f"{hours:02d}:{minutes:02d}:00"
        
        # Node-specific parameters
        if node_type == 'gpu':
            partition = 'gpu'
            gpu_param = f"--gpus-per-node={gpus}" if gpus else "--gpus-per-node=1"
        elif node_type == 'large':
            partition = 'bigmem'
            gpu_param = ""
        else:
            partition = 'standard'
            gpu_param = ""
        
        params = {
            'sbatch_command': f"sbatch --nodes={nodes} --partition={partition} --time={time_str} {gpu_param}",
            'partition': partition,
            'time': time_str,
            'nodes': nodes
        }
        
        return params
    
    def visualize_scaling(self, task_spec, max_nodes=32):
        """
        Visualize task scaling with different node counts
        """
        node_type = self.recommend_hpc_resources(task_spec)['node_type']
        node_specs = self.hpc_configs[node_type]
        
        # Calculate estimated runtime for different node counts
        nodes = list(range(1, max_nodes + 1))
        runtimes = []
        
        for n in nodes:
            resources = {
                'cpu_cores': n * node_specs['cores_per_node'],
                'memory_gb': n * node_specs['memory_per_node']
            }
            runtime = self._estimate_runtime_with_resources(task_spec, resources)
            runtimes.append(runtime)
        
        # Ideal scaling (Amdahl's Law limit)
        serial_fraction = 1.0 - (task_spec['parallelizable_percentage'] / 100.0)
        ideal_scaling = [
            task_spec['base_runtime_hours'] * (
                serial_fraction + (1 - serial_fraction) / (n * node_specs['cores_per_node'] / task_spec['required_cores'])
            )
            for n in nodes
        ]
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(nodes, runtimes, 'b-', marker='o', label='Estimated Runtime')
        plt.plot(nodes, ideal_scaling, 'g--', label='Theoretical Limit')
        plt.axhline(y=task_spec['base_runtime_hours'], color='r', linestyle=':', label='Base Runtime')
        
        # Format
        plt.xlabel('Number of Nodes')
        plt.ylabel('Runtime (hours)')
        plt.title(f'Task Scaling: {task_spec.get("name", "Task")} ({task_spec["task_type"].value})')
        plt.legend()
        plt.grid(True)
        
        # Cost-efficiency analysis
        cost_per_node = node_specs['cost_per_node_hour']
        total_cost = [n * cost_per_node * runtime for n, runtime in zip(nodes, runtimes)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(nodes, total_cost, 'r-', marker='o')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Total Cost')
        plt.title('Cost Analysis')
        plt.grid(True)
        
        # Find optimal node count for cost efficiency
        min_cost_index = total_cost.index(min(total_cost))
        plt.axvline(x=nodes[min_cost_index], color='g', linestyle='--', 
                   label=f'Optimal: {nodes[min_cost_index]} nodes')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'optimal_nodes': nodes[min_cost_index],
            'optimal_runtime': runtimes[min_cost_index],
            'optimal_cost': total_cost[min_cost_index]
        }

# Example usage
if __name__ == "__main__":
    calculator = ResourceCalculator()
    
    # Example task specification
    clustering_task = {
        'name': 'Entity Clustering Analysis',
        'base_runtime_hours': 4.0,              # Baseline runtime with ideal resources
        'required_memory_gb': 64,               # Memory requirement
        'required_cores': 8,                    # Ideal number of cores
        'task_type': TaskType.CPU_BOUND,        # Bottleneck type
        'parallelizable_percentage': 80,        # Percentage that can be parallelized
        'io_intensity': 0.3                     # I/O intensity (0-1)
    }
    
    # Local resources
    local_resources = {
        'memory_gb': 32,
        'cpu_cores': 4,
        'io_speed_mbps': 500
    }
    
    # Calculate estimated time with local resources
    time_estimate = calculator.calculate_completion_time(clustering_task, local_resources)
    print(f"Estimated completion time: {time_estimate['estimated_hours']:.2f} hours")
    print(f"Limiting factor: {time_estimate['limiting_factor']}")
    
    # Determine if HPC resources should be requested
    hpc_decision = calculator.should_request_hpc(clustering_task, local_resources, deadline_hours=12)
    print(f"Request HPC: {hpc_decision['request_hpc']}")
    print(f"Reason: {hpc_decision['reason']}")
    print(f"Urgency: {hpc_decision['urgency']}")
    
    # If HPC is recommended, get optimal resource configuration
    if hpc_decision['request_hpc']:
        hpc_resources = calculator.recommend_hpc_resources(clustering_task, deadline_hours=12)
        print("\nRecommended HPC configuration:")
        print(f"Node type: {hpc_resources['node_type']}")
        print(f"Nodes required: {hpc_resources['nodes_required']}")
        print(f"Total cores: {hpc_resources['total_cores']}")
        print(f"Total memory: {hpc_resources['total_memory_gb']} GB")
        print(f"Estimated runtime: {hpc_resources['estimated_runtime_hours']:.2f} hours")
        print(f"Cost estimate: ${hpc_resources['cost_estimate']:.2f}")
        print(f"SLURM command: {hpc_resources['slurm_parameters']['sbatch_command']}")
    
    # Visualize scaling with different node counts
    scaling_analysis = calculator.visualize_scaling(clustering_task)
    print(f"\nOptimal configuration for cost efficiency:")
    print(f"Optimal nodes: {scaling_analysis['optimal_nodes']}")
    print(f"Runtime with optimal configuration: {scaling_analysis['optimal_runtime']:.2f} hours")
    print(f"Cost with optimal configuration: ${scaling_analysis['optimal_cost']:.2f}")