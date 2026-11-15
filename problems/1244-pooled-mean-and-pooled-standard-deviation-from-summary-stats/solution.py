def pooled_stats(m1, s1, n1, m2, s2, n2):
    # m1, s1, n1: mean, sample sd, size of group 1
    # m2, s2, n2: mean, sample sd, size of group 2
    # return (combined_mean, combined_sd)
    combined_mean = (m1*n1+m2*n2)/(n1+n2)
    ss_within = (n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)
    ss_between = n1* ((m1 - combined_mean)**2) + n2* ((m2 - combined_mean)**2)
    std = ((ss_between + ss_within) / (n1 + n2 - 1))**0.5
    return combined_mean, std