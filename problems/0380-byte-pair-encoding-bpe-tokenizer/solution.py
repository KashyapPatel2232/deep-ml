import torch

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus using PyTorch for pair counting.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    output = []
    
    def output_pairs(corpus: dict, num_merges: int):
        pair_list = []
        weights = []
        
        for key, freq in corpus.items():
            ls = key.split(" ")
            for i in range(len(ls) - 1):
                pair_list.append((ls[i], ls[i+1]))
                weights.append(freq)
                
        if not pair_list:
            return {}

        unique_pairs = sorted(list(set(pair_list)))
        pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}
        
        indices = torch.tensor([pair_to_idx[p] for p in pair_list], dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.long)

        counts = torch.bincount(indices, weights=weights_tensor)
        
        sorted_indices = torch.argsort(counts, descending=True, stable=True)
        
        pairs = {
            unique_pairs[idx.item()]: counts[idx.item()].item() 
            for idx in sorted_indices
        }
        return pairs

    new_corpus = corpus
    for _ in range(num_merges):
        combinations = output_pairs(new_corpus, num_merges)
        if not combinations: 
            break
            
        most_freq = list(combinations.keys())[0]
        output.append(most_freq)
        
        updated_corpus = {}
        for key in new_corpus:
            freq = new_corpus[key]
            ls = key.split(" ")
            temp = []
            
            j = 0
            while j < len(ls):
                if j < len(ls) - 1 and (ls[j], ls[j+1]) == most_freq:
                    temp.append(ls[j] + ls[j+1])
                    j += 2  
                else:
                    temp.append(ls[j])
                    j += 1 
                    
            updated_corpus[" ".join(temp)] = freq
            
        new_corpus = updated_corpus

    return output