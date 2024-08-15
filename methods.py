import numpy as np
from scipy.linalg import sqrtm
import itertools as it

tolerance = 1e-4


def triangle_inequality(A, B, length, positive) -> bool:
    difference = length(A) + length(B) - length(A + B)
    if positive(difference):
        return True
    else:
        return False


def is_significant(a) -> bool:
    if np.all(abs(a) < tolerance):
        return False
    else:
        return True


def is_hermitian(A) -> bool:
    if is_significant(A.conj().T - A):
        return False
    else:
        return True


def is_normal(A):
    if is_significant(A.conj().T.dot(A) - A.dot(A.conj().T)):
        return False
    else:
        return True


def is_commuting(A, B):
    if is_significant(A.dot(B) - B.dot(A)):
        return False
    else:
        return True


def spectral_decomposition(A):
    if not is_hermitian(A):
        raise ValueError("Spectral Decomposition: A is not Diagonalizable")
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        D = np.diag(eigenvalues)
        P = eigenvectors
        return P, D


def matrix_absolute_value_scipy(A):
    AHA = A.conjugate().transpose().dot(A)
    return sqrtm(AHA).astype(
        np.complex128
    )  # The scipy method returns complex256 data. Which does not work with numpy.linalg.


def matrix_absolute_value(A):
    AHA = A.conjugate().transpose().dot(A)
    P, D = spectral_decomposition(AHA)
    diagonal_roots = np.sqrt(np.absolute(np.diag(D)))
    D = np.diag(diagonal_roots)
    return P.dot(D).dot(P.conj().T)


def is_psd(A) -> bool:
    if not is_hermitian(A):
        return False
    min_eigenvalue = np.min(np.linalg.eigvalsh(A))
    if not is_significant(min_eigenvalue):
        return True
    elif min_eigenvalue > 0:
        return True
    else:
        return False


def is_pd(A) -> bool:
    if not is_hermitian(A):
        return False
    min_eigenvalue = np.min(np.linalg.eigvalsh(A))
    if not is_significant(min_eigenvalue):
        return False
    elif min_eigenvalue > 0:
        return True
    else:
        return False


def is_psd_cholesky(A) -> bool:  # Not Working
    if not is_hermitian(A):
        return False
    else:
        try:
            L = np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False


def is_psd_trace_det(A) -> bool:
    if not is_hermitian(A):
        return False
    if np.shape(A) != (2, 2):
        raise ValueError(
            "Matrix Absolute Value (Tr/Det Method): Matrix has dimension greater than 2"
        )
    tr = A[0][0] + A[1][1]
    det = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])
    if is_significant(tr) and is_significant(det):
        if tr > 0 and det > 0:
            return True
    elif is_significant(tr) and not is_significant(det):
        if tr > 0:
            return True
    elif not is_significant(tr) and not is_significant(det):
        return True
    else:
        return False


def M_n(min, max, n, type="Integer"):
    if type == "Integer":
        for elms in it.product(range(min, max + 1), repeat=n**2):
            yield np.reshape(elms, (n, n))
    elif type == "Gaussian":
        for elms in it.product(range(min, max + 1), repeat=2 * (n**2)):
            yield np.array(
                [z[0] + 1j * z[1] for z in np.reshape(elms, (2 * n, 2))]
            ).reshape((n, n))


def random_matrix(n, min, max, type="Integer", count=1):
    samples = []
    for _ in range(count):
        if type == "Gaussian Integer":
            A = []
            for _ in range(n**2):
                a, b = np.random.randint(min, max + 1, size=2)
                A.append(a + 1.0j * b)
            samples.append(np.array(A, dtype=np.complex128).reshape((n, n)))
        elif type == "Complex":
            A = []
            for _ in range(n**2):
                a, b = np.random.uniform(min, max + 1, size=2)
                A.append(a + 1.0j * b)
            samples.append(np.array(A, dtype=np.complex128).reshape((n, n)))
        elif type == "Real":
            samples.append(np.random.uniform(min, max + 1, (n, n)))
        elif type == "Integer":
            samples.append(np.random.randint(min, max + 1, (n, n)))
        else:
            raise ValueError(f"random_matrix: {type} is not a valid type.")
    return samples


def triangle_inequality_failure_rate(n, a):
    matrices = list(M_n(-a, a, n))
    sucess_count = -len(matrices)
    total_pairs = -len(matrices)
    num_matrices = len(matrices)
    for i in range(num_matrices):
        for j in range(i, num_matrices):
            if triangle_inequality(
                matrices[i], matrices[j], matrix_absolute_value, is_psd
            ):
                sucess_count += 2
            total_pairs += 2
    success_rate = sucess_count / total_pairs if total_pairs else 0
    failure_rate = 1 - success_rate
    return failure_rate


if __name__ == "__main__":
    entry_ranges = [1, 2, 3, 4, 5]
    for type in [
        "Gaussian Integer",
        "Complex",
    ]:
        percentages = []
        for e in entry_ranges:
            sample_size = 100_000
            failed = 0
            for sample in range(sample_size):
                A, B = random_matrix(n=2, min=-e, max=e, type=type, count=2)
                if not triangle_inequality(
                    A,
                    B,
                    length=matrix_absolute_value_scipy,
                    positive=is_psd,
                ):
                    failed += 1
            percentages.append(failed / sample_size)
