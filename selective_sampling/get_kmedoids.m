function get_kmedoids(featurefile, k)
    % Select samples using medoids
    %
    % Args:
    %     featurefile: Filename of file containing the embeddings of samples (Nfiles, Ndims)
    %     k (int): Number of medoids to consider (equal to number of selected samples)
    %
    % Returns:
    %    selected_indices (array): Indices of selected samples

    % Load feature
    features = load(featurefile);
    X = features.features;

    [idx,C] = kmedoids(X, k, 'Algorithm', 'pam', 'Distance', 'cosine', 'Options', statset('MaxIter',300), 'Replicates', 10);
    medoids = find(ismember(X, C, 'rows'));
    
    save("medoids.mat", "medoids");
end