import pandas as pd
import numpy as np
import networkx as nx
from os import listdir
from os.path import isfile, join
import sys


def level_calc(hc_location):
    """Calculate the dependency levels of the base TSVs to provide the
    proper order for running tsv_dist. Tony's code.

    Args:
        hc_location (str): location of the base TSVs (suggested form
            'housing_characteristics/hc1')

    Returns:
        level_dct (dct): dictionary of all of the TSVs by level (keys
            are level and values are lists of TSVs at each level.
        adj_df (pandas.DataFrame): dependency matrix of TSVs
    """

    # Load the file names into memory
    HC_files = [f for f in listdir(hc_location) if isfile(join(hc_location, f))]
    # Ignore any hidden files (beginning with ".")
    idx = []
    for i in range(len(HC_files)):
        if HC_files[i][0] != ".":
            idx.append(i)
    HC_files = list(np.array(HC_files)[idx])

    # Remove the .tsv from the housing characteristic name
    HC_names = HC_files.copy()
    i = 0
    for HC_str in HC_names:
        HC_names[i] = HC_str.split('.')[0]
        i += 1
        # Initialize the adjacency matrix
    adj_mat = np.zeros((len(HC_names), len(HC_names)))

    # For each housing characteristic
    for i in range(len(HC_files)):

        # Read the first line of the housing characteristic file
        with open(hc_location + '/' + HC_files[i]) as f:
            header_str = f.readline()

        # For each column in the tsv file
        for column_name in header_str.split('\t'):

            # If there is a dependency
            if len(column_name) > 0:
                if column_name[0] == 'D':
                    if column_name.find('Dependency='):
                        print(HC_files[i], column_name)

                    # Get the dependency name
                    dependency_str = column_name.split('=')[1]

                    # Find in the housing characteristics names
                    j = HC_names.index(dependency_str)

                    # Include the depenancy in the adjacency matrix
                    adj_mat[i, j] = 1

    # Convert to Pandas
    adj_df = pd.DataFrame(adj_mat, index=HC_names, columns=HC_names).T
    # Create Graph
    G = nx.from_pandas_adjacency(adj_df, nx.DiGraph())
    n_nodes = len(G.nodes)
    node_names = adj_df.columns.values
    longest_path_len = nx.dag_longest_path_length(G)
    # print(nx.info(G))
    # print('Longest path length:\t %d' % longest_path_len)

    # Initialize node level values
    level = np.zeros_like(adj_df[adj_df.columns[0]])

    # Iterate over nodes up to the longest path length
    for j in range(int(longest_path_len) + 1):
        # For each housing characteristic node
        for i in range(n_nodes):
            # Get the index of the dependencies
            column_name = adj_df.columns[i]
            dependencies = np.where(adj_df[column_name] == 1)[0]

            if len(dependencies) > 0:  # If any dependencies
                # Get the level of the dependencies
                dependency_levels = level[dependencies]

                # Identify the maximum level of the dependencies
                max_level = np.max(dependency_levels)

                # This node's level must be one greater than the max level
                level[i] = max_level + 1
    level_dct = {}
    for i in range(int(longest_path_len) + 1):
        idx = np.where(level == i)[0]
        nam = []
        for j in range(len(idx)):
            nam.append(node_names[idx[j]])
        level_dct[i] = nam
    return level_dct, adj_df


def tsv_process(tsv_name, curr_path):
    """Opens a TSV, normalizes by ResStock Weights and returns the weighted DF
    as well as a list of the dependencies in that TSV

    Args:
        tsv_name (str): name of the tsv (w/o extension)
        curr_path (str): location of the tsv

    Returns:
        dep_df (pandas.frame.DataFrame): Dataframe with the "Options="
            columns weighted by the ResStock Weight (ResStock weight
            column is removed)
        deps (list): list of the dependencies within that tsv. Listed w/ the
            'Dependency=' label.
    """
    try:
        dep_df = pd.read_csv(
            curr_path + '/' + tsv_name + '.tsv', delimiter='\t', comment='#')
    except FileNotFoundError:
        print(curr_path + '/' + tsv_name, 'file not found')

    opts = [a for a in dep_df.columns if a.startswith('Option=')]
    deps = [a for a in dep_df.columns if a.startswith('Dependency=')]
    cols_to_keep = opts + deps + ['resstock_probability']
    dep_df = dep_df[cols_to_keep]
    dep_df.loc[:, dep_df.columns.str.startswith('Option=')] = dep_df.loc[:, dep_df.columns.str.startswith(
        'Option=')].multiply(dep_df['resstock_probability'], axis=0)
    dep_df = dep_df.drop(['resstock_probability'], axis=1)
    return dep_df, deps


def tsv_dist(tsv_name, curr_path, write_tf=True):
    """Add or update column in TSVs corresponding to row selection probability

    This function calculates and inserts in each TSV and its dependent
    TSVs a column of weights that indicate the joint probability of a
    TSV row being selected in ResStock when accounting for the
    dependencies from other TSVs. This value is called the ResStock
    Weight, as indicated in the heading for the corresponding column.

    This function must be called in "Level" order starting with the
    TSVs that have no dependencies and working up to the TSVs with the
    most dependencies. This function should be used in conjunction with
    the 'level_calc' function to determine the run order, as this might
    change as the TSV dependency structure is updated. Running out of
    level order might give incorrect weights. 'tsv_update' will run this
    function for the full set of TSVs.

    Args:
        tsv_name (str): The technology name for a TSV (e.g., "Clothes
            Washer" for the Clothes Washer.tsv file)
        curr_path (str): The path for the housing characteristics
            folder for the current scenario and interval
        write_tf (bool): Should the function write the tsv

    Returns:
        A series with the probability of each TSV option being selected
        The weights for the updated TSV are exported to the file directly.
    """
    # Import the main TSV of interest and pre-process the dataframe

    level_dct, adj_df = level_calc(curr_path)
    print(tsv_name)

    try:
        file_path = curr_path + '/' + tsv_name + '.tsv'
        df = pd.read_csv(file_path, delimiter='\t', comment='#')
    except FileNotFoundError:
        # If the file is not available in the current directory,
        # exit the function without attempting to recalculate
        # the ResStock_Weight values; otherwise, this function
        # has a tendency to overwrite the weight values correctly
        # calculated for previous intervals weights for the
        # current interval
        return

    dependencies_ = [a for a in df.columns if a.startswith('Dependency=')]
    dependencies = [a.split('=')[1] for a in dependencies_]
    cols_to_keep = [a for a in df.columns if a.startswith('Option=')] + dependencies_
    df = df[cols_to_keep]

    # Determine the option columns
    option_cols = []
    for col in df.columns.values:
        if ("Option=" in col):
            option_cols.append(col)

    # Start weight calculations
    # If there are dependencies (almost all dataframes)
    if dependencies_:
        df[dependencies_] = df[dependencies_].astype('str')
        # Convert the dependencies into a MultiIndex
        df = df.set_index(dependencies_)
        # Remove "Option=" from the column names
        df.columns = [col[7:] for col in df.columns]

        # Initialize a dict to store the upstream dependencies
        # as lists for each dependency in a TSV
        dep_set = {}

        # Instantiate a short list of the number of overlapping
        # dependencies and the TSV that contains them
        dep_lst = [0, '']
        for dep in dependencies:
            # Find upstream dependencies
            dep_set[dep] = set(list(adj_df[adj_df[dep] != 0][dep].index.values) + [dep])
            # Find the number of dependencies in the intersection
            a = len(set(dependencies) & set(dep_set[dep]))
            # Check to see if there are more dependencies than
            # previously checked dependencies
            if a > dep_lst[0]:
                dep_lst = [a, dep]  # Put the new max length into the list
        # print(dep_set[dep])

        # Case for conditional probabilities
        if dep_lst[0] > 1:
            # Allows TSVs with conditional probabilities to be identified
            # print(tsv_name,dep_lst)

            # Find any non-conditional probabilities
            non_cond = set(dependencies) - dep_set[dep_lst[1]]

            # Get the upstream TSV weighted by the ResStock weights
            dep_df, deps = tsv_process(dep_lst[1], curr_path)
            deps_ = set([x[11:] for x in deps])  # Strip off 'Dependency='

            # Identify columns to drop - dependencies in the upstream
            # TSV that are not also in the dataframe
            drops = deps_ - set(dependencies) & deps_
            drops = ['Dependency=' + x for x in list(drops)]
            dep_df = dep_df.drop(drops, axis=1)

            # Group by the dependencies of interest in the upstream TSV
            dep_df = dep_df.groupby(list(set(dependencies_) & set(deps))).sum()

            # Strip off "Option="
            dep_df.columns = [col[7:] for col in dep_df.columns]

            # Transpose the dependency dataframe such that the
            # options become a new index level
            dep_df = pd.melt(dep_df.reset_index(),
                             id_vars=dep_df.index.names,
                             var_name='Dependency=' + dep_lst[1],
                             value_name='resstock_probability')

            # For the case where there are both conditional
            # probabilities and non-conditional probabilities
            if non_cond:
                holding = {}
                # Loop over the non-conditional dependencies
                for ele in non_cond:
                    # Pull the tsvs for the dependencies
                    ind_df, deps = tsv_process(ele, curr_path)
                    # Process each of the tsvs and store in a holding list
                    if deps:
                        ind_df = ind_df.set_index(deps)
                    ind_df.columns = [col[7:] for col in ind_df.columns]
                    new_levs = ind_df.columns
                    ind_df = ind_df.sum()
                    holding[ele] = ind_df
                    # Extend the dep_df into which the mutually
                    # exclusive probabilities will be joined
                    b = len(dep_df)
                    dep_df = pd.concat([dep_df] * len(new_levs))
                    dep_df["Dependency=" + ele] = np.repeat(new_levs, b)

                # Update index to match main dataframe (order is important)
                dep_df = dep_df.reset_index().set_index(df.index.names)
                # Remove index column from reindexing process
                dep_df = dep_df.drop(['index'], axis=1)
                # Multiply the mutually exclusive probabilities
                # into the dependency dataframe
                for ele in holding:
                    dep_df = dep_df.multiply(holding[ele], axis=0, level='Dependency=' + ele)
            else:
                # Update index to match main dataframe (order is important)
                dep_df = dep_df.set_index(df.index.names)

            # Update main dataframe with the resultant weights
            df = df.sort_index()
            dep_df = dep_df.sort_index()
            df['resstock_probability'] = np.array(dep_df['resstock_probability'])

        # If all probabilities are mutually exclusive (dep_lst[0] should == 1)
        else:
            # a = []
            # Go through mutually exclusive dependencies and pull the weights
            for fle in dependencies:
                dep_df, deps = tsv_process(fle, curr_path)
                try:
                    dep_df = dep_df.set_index(deps)
                    dep_df.columns = [col[7:] for col in dep_df.columns]
                    try:
                        df = df.multiply(dep_df.sum(), axis=0, level='Dependency=' + fle)
                    except Exception:
                        df = df.multiply(np.array(dep_df.sum()), axis=0)
                except Exception:
                    dep_df.columns = [col[7:] for col in dep_df.columns]
                    dep_df = pd.melt(dep_df, var_name='Dependency=' + fle,
                                     value_name='resstock_probability')
                    dep_df = dep_df.set_index('Dependency=' + fle)
                    try:
                        df = df.multiply(dep_df.sum(axis=1), axis=0, level='Dependency=' + fle)
                    except Exception:
                        df = df.multiply(dep_df.sum(axis=1), axis=0)
            df['resstock_probability'] = df.sum(axis=1)

    # For dataframes with no dependencies
    else:
        df['resstock_probability'] = 1

    # Update tsv
    final_df = pd.read_csv(file_path, delimiter='\t', comment='#')

    # Update formatting of final dataframe
    cols_to_look_for = ['ResStock_Probability',
                        'Marginal_Probability',
                        'resstock_probability'
                        'ResStock_Weight',
                        'Manual',
                        'Manual Weights',
                        'Manual Weight',
                        '[For Reference Only] ResStock_Weight']  # used in function testing

    cols_to_remove = [a for a in final_df.columns if a in cols_to_look_for]
    final_df = final_df.drop(cols_to_remove, axis=1)

    # Add marginal probability column
    if dependencies_:
        final_df[dependencies_] = final_df[dependencies_].astype('str')
        final_df = final_df.set_index(dependencies_)
        final_df['resstock_probability'] = df['resstock_probability']
        if write_tf:
            final_df.to_csv(file_path, sep='\t')
    else:
        final_df['resstock_probability'] = 1
        if write_tf:
            final_df.to_csv(file_path, sep='\t', index=False)

    final_df[option_cols] = final_df[option_cols].multiply(df['resstock_probability'], axis=0)

    options = [col[7:] for col in final_df[option_cols].columns.values]
    for col in option_cols:
        final_df.rename(columns={col: col[7:] for col in option_cols}, inplace=True)
    return final_df.reset_index()[options].sum(axis=0)


def children(tsv_name, curr_path):
    """ For a tsv, returns all downstream dependencies in a dictionary with
    the level as the key and the downstream tsvs as the value. Used for speed
    improvement in tsv_update

    Args:
        tsv_name (str): name of the tsv (w/o extension)
        curr_path (str): location of the tsv

    Returns:
        child_level (dct): dictionary of downstream dependencies with level
            as key and dependencies at that level as value
    """

    level_dct, adj_df = level_calc(curr_path)

    def child_list(tsv_name, curr_path):
        """ For a tsv or set of tsvs, returns a list of the downstream
        tsvs that call the initial tsv(s) as a dependency

        Args:
            tsv_name (str or set): name(s) of the tsv (w/o extension)
            curr_path (str): location of the tsv

        Returns:
            tsv_name (set): set of tsvs that call the initial
                tsvs as dependency
        """
        if not isinstance(tsv_name, set):
            tsv_name = {tsv_name}
        tsv_name = set([i for l in [adj_df.loc[adj_df.loc[ele] == 1][ele].axes[0].tolist()
                                    for ele in tsv_name] for i in l])
        return tsv_name

    full_set = {}
    child_level = {}
    for i in range(max(level_dct.keys()) + 1):
        child_level[i] = []
    while tsv_name:
        tsv_name = child_list(tsv_name, curr_path)
        full_set = tsv_name.union(full_set)
    for ele in full_set:
        lev = [level for level, tsvs in level_dct.items() if ele in tsvs][0]
        child_level[lev] = child_level[lev] + [ele]
    return child_level


def tsv_update(curr_path, level_dct, level_look='default'):
    """
    This function updates all tsvs within a given folder with their joint
    probability of being selected. Must correspond with level_dct mapping.

    Args:
        curr_path (str): TSV folder

        level_look (dct): optional argument for where to look for the
            levels to process. Default is full set of TSVs


    Returns:
        Nothing. TSVs have their ResStock Weight column updated.

    """

    if level_look == 'default':
        level_look = level_dct
    else:
        level_look = children(level_look, curr_path)
    lev_max = max(level_look.keys())
    for lev in range(lev_max + 1):
        [tsv_dist(x, curr_path) for x in level_look[lev]]


if __name__ == '__main__':
    path_HCs = sys.argv[1]  # '../housing_characteristics/'
    #path_HCs = 'housing_characteristics/'
    level_dct, adj_df = level_calc(path_HCs)
    tsv_update(path_HCs, level_dct)
