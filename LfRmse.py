# linear functional conditional expectation convergence counter example
import numpy as np
from MyGPR import RBF, Matern32, Matern52
from VariableTrainSetGPR import ToeplitzGPR
import matplotlib.pyplot as plt
import dill
import os


def LfRmse(k, dr_id, finiteDiff, diffdelta, NX):
    """
    Compute conditional L(f)|D for different kernels k and linear operators L
    
    dr_id, finiteDiff, diffdelta specify L, e.g.:
        dr_id = 2
        finiteDiff = False
        diffdelta = .05
        k = RBF(1,1,1)
    """

    LB = -0.5
    nsvar = .01
    NREP = 5000

    ZERO = np.zeros((1,1))
    drtypes = [(), (0,), (0,0)]

    if not finiteDiff:
        diffdelta = None

    def calc_KTsTr():
        if not finiteDiff:
            KTsTr = k.K(ZERO, X, drtypes[dr_id])[1]
        else:
            assert diffdelta > 0
            KTsTr = (k.K(ZERO + diffdelta/2., X, drtypes[dr_id])[1] -
                     k.K(ZERO - diffdelta/2., X, drtypes[dr_id])[1]) / diffdelta
        return KTsTr

    X = np.linspace(LB, -LB, NX).reshape((-1,1))
    Y = np.random.normal(size=(NX,NREP))*np.sqrt(nsvar)
    KTsTr = calc_KTsTr()
    mod = ToeplitzGPR(X, k, nsvar, KTsTr)
    prd = mod.predict(Y).flatten()
    rmse = np.sqrt(np.mean(prd**2))
    return rmse


def run_experiment_and_save_results():
    if os.path.exists('out/LfRmse_results.pkl'):
        # results already computed
        return

    NXs = np.array([20,100,500,1000,2000,6000,10000,20000,40000])+1
    diffdeltas = [.16,.11,.08,.06,.04,.03,.02]
    rmses_m32_Li = [np.full(NXs.shape, np.nan) for _ in diffdeltas]
    rmses_m32_L = np.full(NXs.shape, np.nan)

    print('k=Matern32, run Li experiments')
    for j, diffdelta in enumerate(diffdeltas):
        print(j, f'delta={diffdelta}')
        for i, NX in enumerate(NXs):
            rmses_m32_Li[j][i] = LfRmse(Matern32(1,1,1), dr_id=1, finiteDiff=True, diffdelta=diffdelta, NX=NX)

    print('k=Matern32, run L experiment')
    for i, NX in enumerate(NXs):
        rmses_m32_L[i] = LfRmse(Matern32(1,1,1), dr_id=2, finiteDiff=False, diffdelta=None, NX=NX)

    rmses_rbf_Li = [np.full(NXs.shape, np.nan) for _ in diffdeltas]
    rmses_rbf_L = np.full(NXs.shape, np.nan)

    print('k=RBF, run Li experiments')
    for j, diffdelta in enumerate(diffdeltas):
        print(j, f'delta={diffdelta}')
        for i, NX in enumerate(NXs):
            rmses_rbf_Li[j][i] = LfRmse(RBF(1,1,1), dr_id=1, finiteDiff=True, diffdelta=diffdelta, NX=NX)

    print('k=RBF, run L experiment')
    for i, NX in enumerate(NXs):
        rmses_rbf_L[i] = LfRmse(RBF(1,1,1), dr_id=2, finiteDiff=False, diffdelta=None, NX=NX)

    print('Save results')
    os.makedirs('out', exist_ok=True)
    with open('out/LfRmse_results.pkl','wb') as f:
        dill.dump([NXs, diffdeltas, rmses_m32_Li, rmses_m32_L, rmses_rbf_Li, rmses_rbf_L], f)


def load_and_plot_results():
    with open('out/LfRmse_results.pkl','rb') as f:
        NXs, diffdeltas, rmses_m32_Li, rmses_m32_L, rmses_rbf_Li, rmses_rbf_L = dill.load(f)

    fig, ax = plt.subplots(1,1,figsize=(5.5,4.8))
    ax.set_prop_cycle(linestyle=['--','-.',':','--','-.',':','--'], marker=['+','o','^','s','x','>','x'])
    for i, rmse in enumerate(rmses_m32_Li):
        ax.semilogx(NXs, rmse, color='gray', label=f'$\epsilon_i$={diffdeltas[i]:.2f}')
    ax.semilogx(NXs, rmses_m32_L, 'k-o', label='L')
    ax.legend()
    ax.set_xlabel('$|D_n|$', fontsize=13)
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_title('Matern32', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1,1,figsize=(5.5,4.8))
    ax.set_prop_cycle(linestyle=['--','-.',':','--','-.',':','--'], marker=['+','o','^','s','x','>','x'])
    for i, rmse in enumerate(rmses_rbf_Li):
        ax.semilogx(NXs, rmse, color='gray', label=f'$\epsilon_i$={diffdeltas[i]:.2f}')
    ax.semilogx(NXs, rmses_rbf_L, 'k-o', label='L')
    ax.legend()
    ax.set_xlabel('$|D_n|$', fontsize=13)
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_title('RBF', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    fig.show()
    

if __name__ == "__main__":
    run_experiment_and_save_results()
    load_and_plot_results()
    _ = input('Press Enter to exit')
