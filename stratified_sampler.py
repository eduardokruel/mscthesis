import re
from collections import defaultdict

def identify_hop_count(example_id):
    """
    Identify the hop count from an example ID.
    
    Args:
        example_id: The ID of the example (e.g., "2hop__123456_789012")
        
    Returns:
        int: The number of hops (2, 3, or 4) or None if not identifiable
    """
    if isinstance(example_id, str):
        # Try to extract hop count from the ID format
        match = re.match(r'(\d+)hop', example_id)
        if match:
            return int(match.group(1))
    
    return None

def deterministic_stratified_sample(df, batch_size):
    """
    Create a deterministic stratified sample of examples based on hop counts.
    This preserves the original order within each hop group.
    
    Args:
        df: DataFrame containing the dataset
        batch_size: Total number of examples to sample
        
    Returns:
        list: List of example indices to process
    """
    # Group examples by hop count while preserving original order
    hop_groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        example_id = row.get('id')
        hop_count = identify_hop_count(example_id)
        if hop_count:
            hop_groups[hop_count].append(idx)
        else:
            # For examples without identifiable hop count, put in a separate group
            hop_groups['unknown'].append(idx)
    
    # Calculate target counts for each hop group
    total_examples = len(df)
    target_counts = {}
    remaining = batch_size
    
    # Calculate proportions for each hop count in the original dataset
    proportions = {hop: len(indices) / total_examples for hop, indices in hop_groups.items()}
    
    # Calculate how many examples to sample from each group
    for hop, prop in proportions.items():
        # Calculate the number of examples to sample from this group
        count = int(batch_size * prop)
        target_counts[hop] = count
        remaining -= count
    
    # Distribute any remaining examples due to rounding
    if remaining > 0:
        # Sort hops by their fractional part to distribute remaining examples fairly
        fractional_parts = [(hop, batch_size * prop - target_counts[hop]) 
                           for hop, prop in proportions.items()]
        fractional_parts.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(remaining):
            hop = fractional_parts[i % len(fractional_parts)][0]
            target_counts[hop] += 1
    
    # Interleave examples from different hop groups to create a balanced sample
    # while preserving order within each group
    sampled_indices = []
    group_positions = {hop: 0 for hop in hop_groups.keys()}
    
    # Continue until we've filled our batch
    while len(sampled_indices) < batch_size:
        # Try to take one example from each hop group in turn
        for hop in sorted(hop_groups.keys()):
            # Skip if we've already taken enough from this group
            if len([idx for idx in sampled_indices if idx in hop_groups[hop]]) >= target_counts[hop]:
                continue
                
            # Skip if we've exhausted this group
            if group_positions[hop] >= len(hop_groups[hop]):
                continue
                
            # Take the next example from this group
            sampled_indices.append(hop_groups[hop][group_positions[hop]])
            group_positions[hop] += 1
            
            # Stop if we've reached our target
            if len(sampled_indices) >= batch_size:
                break
    
    # Ensure we don't exceed the batch size
    return sampled_indices[:batch_size]

def stratified_sample(df, batch_size):
    """
    Wrapper function that calls the deterministic version for backward compatibility.
    """
    return deterministic_stratified_sample(df, batch_size)

def get_hop_distribution(df, indices=None):
    """
    Get the distribution of hop counts in the dataset or a subset.
    
    Args:
        df: DataFrame containing the dataset
        indices: Optional list of indices to analyze
        
    Returns:
        dict: Distribution of hop counts
    """
    if indices is not None:
        subset = df.iloc[indices]
    else:
        subset = df
    
    hop_counts = defaultdict(int)
    total = len(subset)
    
    for _, row in subset.iterrows():
        example_id = row.get('id')
        hop_count = identify_hop_count(example_id)
        if hop_count:
            hop_counts[hop_count] += 1
        else:
            hop_counts['unknown'] += 1
    
    # Convert to percentages
    distribution = {hop: (count / total) * 100 for hop, count in hop_counts.items()}
    
    return distribution 