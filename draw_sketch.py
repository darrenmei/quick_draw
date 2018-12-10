# draw image sketch
X_dev_n = X_strokes
print('X_Dev_n: ', X_dev_n.shape)
fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(X_dev_n.shape[0]), size = 3)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = X_dev_n[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
    #print('test arr: ', test_arr)
    lab_idx = np.cumsum(test_arr[:,2]-1)
    #print('lab idx: ', lab_idx)
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0],
                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(Y_train[c_id])
