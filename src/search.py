import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_prob = 1.0

    out = [np.argmax(y_probs[:,0,:])]
    forward_path = [np.argmax(y_probs[:,0,:])]
    forward_prob *= y_probs[out[-1],0,:]/np.sum(y_probs[:,0,:])
    for t in range(1, y_probs.shape[1]):
        out.append(np.argmax(y_probs[:,t,:]))
        forward_prob *= y_probs[out[-1],t,:]/np.sum(y_probs[:,t,:])
        if (out[-1] !=  forward_path[-1]):
            forward_path.append(out[-1])

    # if (out[-1] !=  out[-2]):
    #     forward_prob *= y_probs[out[-1],-1,:]/np.sum(y_probs[:,-1,:])
    #     forward_path.append(out[-1])
    #forward_path = [3, 1, 2, 0, 0, 2]
    ans = []
    for i in forward_path:
        if i != 0:
            ans.append(SymbolSets[i-1])
    #print(ans)
    ans = ''.join(ans)
    #print(ans)

    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    #print(type(forward_prob))

    return (ans, float(truncate(forward_prob.astype(float).item(), 19)))
    #return (ans, truncate(float(forward_prob),18))

##############################################################################

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
def BeamSearch(SymbolSets, y_probs, BeamWidth):
    BeamWidth -= 1
    #print("BeamWidth", BeamWidth)
    #print("y_probs", y_probs.shape)
    PathScore = {}
    BlankPathScore = {}

    def InitializePaths(SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        InitialBlankPathScore[path] = y[0] # Score of blank at t=1
        InitialPathsWithFinalBlank = {path}
        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = {}

        for i,c in enumerate(SymbolSet): # This is the entire symbol set, without the blank
            path = c
            InitialPathScore[path] = y[i+1] # Score of symbol c at t=1
            InitialPathsWithFinalSymbol[path] = 0 # Set addition

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        # print("prune", BlankPathScore)
        # print("PathsWithTerminalBlank",PathsWithTerminalBlank)
        # print("PathsWithTerminalSymbol",PathsWithTerminalSymbol)
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        # First gather all the relevant scores
        scorelist = []
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])

        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])

        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist.sort(reverse = True) # In decreasing order
        #print("score", scorelist)
        cutoff = scorelist[BeamWidth] if BeamWidth < len(scorelist) else scorelist[-1]

        PrunedPathsWithTerminalBlank = {}
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank[p] = 0 # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]
        
        PrunedPathsWithTerminalSymbol = {}
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol[p] = 0 # Set addition
                PrunedPathScore[p] = PathScore[p]

        # print("PrunedBlankPathScore",PrunedBlankPathScore)
        # print("PrunedPathScore", PrunedPathScore)
        # print("PrunedPathsWithTerminalBlank", PrunedPathsWithTerminalBlank)
        # print("PrunedPathsWithTerminalSymbol", PrunedPathsWithTerminalSymbol)
        # print("BlankPathScore", BlankPathScore)
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore
    
    def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        #print("ExtendWithBlank",BlankPathScore)
        UpdatedPathsWithTerminalBlank = {}
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank[path] = 0 # Set addition
            UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]
        
        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path]* y[0]
            else:
                UpdatedPathsWithTerminalBlank[path] = 0 # Set addition
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = {}
        UpdatedPathScore = {}
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                newpath = path + c # Concatenation
                UpdatedPathsWithTerminalSymbol[newpath] = 0 # Set addition
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]
        
        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
            for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                newpath = path if c == path[-1] else path + c # Horizontal transitions don’t extend the sequence
                if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                    UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
                else: # Create new path
                    UpdatedPathsWithTerminalSymbol[newpath] = 0 # Set addition
                    UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
        
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths[p] = 0 # Set addition
                FinalPathScore[p] = BlankPathScore[p]

        return MergedPaths, FinalPathScore

    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0,:])
    #print("0\n",NewPathScore, NewBlankPathScore)
    # Subsequent time steps
    for t in range(1,y_probs.shape[1]):
    # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, 
            NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
    # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t,:])
    # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, 
            SymbolSets, y_probs[:,t,:])

        # print(t)
        # print(PathScore, BlankPathScore)
        # print("new," ,NewPathScore, NewBlankPathScore)

    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore,
                                                      NewPathsWithTerminalSymbol, NewPathScore)

    # print(MergedPaths)
    # print(FinalPathScore)
    # Pick best path
    Probs = list(FinalPathScore.values())
    idx = Probs.index(max(Probs)) # Find the path with the best score
    Paths = list(FinalPathScore.keys())
    path = Paths[idx]

    #print(FinalPathScore[path])

    return (path, FinalPathScore)





