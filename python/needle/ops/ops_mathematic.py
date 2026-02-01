"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    """Element-wise addition: a + b
    """
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """Element-wise addition: a + b.

        Args:
            a (NDArray): ndarray of arbitrary shape.
            b (NDArray): ndarray with same shape as a.

        Returns:
            NDArray: output.
        """
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        return out_grad, out_grad


def add(a: NDArray, b: NDArray) -> NDArray:
    """Element-wise addition of ndarrays a, b. Convenience method for EWiseAdd."""
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    """Addition of an ndarray with a scalar: a + scalar.
    """
    def __init__(self, scalar: float):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


def add_scalar(a: NDArray, scalar: float) -> NDArray:
    """Addition of an ndarray with a scalar. Convenience method for AddScalar."""
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    """Elementwise multiplication of two ndarrays: a * b."""
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """Calculates the elementwise multiplication of two ndarrays.

        Args:
            a (NDArray): ndarray with arbitrary shape.
            b (NDArray): ndarray with same shape as a.

        Returns:
            NDArray: a * b.
        """
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    """Elementwise multiplication of two input tensors. Convenience method for EWiseMul."""
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    """Multiply an ndarray with a scalar: a * scalar"""
    def __init__(self, scalar: float):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor]:
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    """Multiply ndarray with a scalar. Convenience method for MulScalar."""
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """Element-wise power, computes: a ** b, where each element of a[i] is raised
        by its corresponding entry in b[i]: a[i] ** b[i].
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    """Raise a to b, elementwise. Convenience method for EWisePow."""
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a: NDArray, scalar: int) -> NDArray:
    """Raise a to (integer) power. Convenience method for PowerScalar."""
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """Compute element-wise division of two ndarrays with same shape.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    """Divide an ndarray by a scalar (elementwise)."""
    def __init__(self, scalar: float):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """Given an ndarray and `axes` (a tuple with two ints), output a new
    ndarray with the two axes swapped.
    When `axes` is None, this should default to swapping the last two dims,
    aka a "transpose" for 2D matrices.
    Example: for a 2D ndarray `a` with shape=[2, 3], Transpose(axes=(0, 1))(a)
        and Transpose()(a) both transpose the matrix, yielding a ndarray with
        shape=[3, 2].
    >>> import needle as ndl
    >>> a = ndl.Tensor([[1, 2, 3], [4, 5, 6]])
    >>> a
    [[1 2 3]
    [4 5 6]]
    >>> a.shape
    (2, 3)
    >>> out = ndl.transpose(a, axes=(0, 1))
    >>> out
    [[1 4]
    [2 5]
    [3 6]]
    >>> out.shape
    (3, 2)
    """
    def __init__(self, axes: Optional[tuple[int, int]] = None):
        # axes: (int axes1, int axes2)
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a: NDArray, axes: Optional[tuple[int, int]]=None) -> NDArray:
    return Transpose(axes)(a)


class Reshape(TensorOp):
    """Reshapes an input ndarray to a new shape. The semantics of the reshape should
    read / write the elements using C-like index order, with the last axis index
    changing fastest, back to the first axis index changing slowest.
    In other words, this should follow numpy.reshape()'s order='C' semantics (the default),
    as defined in the docs: https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html
    Example:
    >>> import needle as ndl
    >>> a = ndl.Tensor([[1, 2, 3], [4, 5, 6]])
    >>> a
    [[1 2 3]
    [4 5 6]]
    >>> a.shape
    (2, 3)
    >>> out = ndl.reshape(a, (2, 3))
    >>> out
    [[1 2]
    [3 4]
    [5 6]]
    >>> out.shape
    (2, 3)
    """
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a: NDArray, shape: tuple[int, ...]) -> NDArray:
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """Broadcast input ndarray to a new shape.
    Tip: refer to the numpy `broadcast_to()` function:
        https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
    """
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a: NDArray, shape: tuple[int, ...]) -> NDArray:
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    """Sum of array elements along given axes."""
    def __init__(self, axes: Optional[tuple[int, ...]] = None):
        """
        Args:
            axes (Optional[tuple[int, ...]], optional): If given, this specifies a
                list of axes to sum over.
                If None, this defaults to summing along ALL available axes.
                Note: for this assignment, you do not need to support negative indexing,
                    eg axes=(-1,)
                Defaults to None.
        """
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        """Sum of array elements along given axes (controlled by `self.axes`).
        Example:
        >>> import numpy as np; import needle as ndl
        >>> a = np.random.rand(3, 4, 5)  # shape=[3, 4, 5]
        >>> out1 = ndl.Summation().compute(a)  # sum along all axes
        >>> out1
        28.688509845460466
        >>> out1.shape  # scalar
        ()
        >>> ndl.Summation(axes=(0, 1, 2)).compute(a)  # equivalent
        28.688509845460466
        >>> out2 = ndl.Summation(axes=(2,)).compute(a)  # sum along axes=2
        >>> out2
        [[1.53128556 2.70279746 4.089636   2.00335739]
        [1.68750711 2.13931317 2.10233744 2.86519078]
        [3.37100156 2.15960558 1.30421554 2.45691378]]
        >>> out2.shape
        (3, 4)
        >>> ndl.Summation(axes=(0, 2)).compute(a).shape  # sum along axes=0 and 2
        (4,)

        Args:
            a (NDArray): input ndarray to sum.

        Returns:
            NDArray: output.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a: NDArray, axes: Optional[tuple[int, ...]] = None) -> NDArray:
    return Summation(axes)(a)


class MatMul(TensorOp):
    """Calculate matrix multiplication (with broadcast support): a @ b"""
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """Calculate matrix multiplication of input ndarrays a, b.
        For 2D a and b, this is matrix multiplication, with shapes:
            a: shape=[n, m]
            b: shape=[m, k]
            a @ b: shape=[n, k]
        For 3D+ shapes a and b, this should calculate "broadcasted"
        matrix mult, where the matmult occurs on the last two dimensions.
        Follows numpy broadcasting semantics.
        Examples:
            a: shape=[3, n, m]
            b: shape=[3, m, k]
            a @ b: shape=[3, n, k]
                Here, three different matmults happen:
                    (a @ b)[0, :, :] = a[0, :, :] @ b[0, :, :]
                    (a @ b)[1, :, :] = a[1, :, :] @ b[1, :, :]
                    (a @ b)[2, :, :] = a[2, :, :] @ b[2, :, :]
            a: shape=[3, n, m]
            b: shape=[   m, k]
            a @ b: shape=[3, n, k]
                Here, the `b` portion is repeated (broadcasted):
                    (a @ b)[0, :, :] = a[0, :, :] @ b[:, :]
                    (a @ b)[1, :, :] = a[1, :, :] @ b[:, :]
                    (a @ b)[2, :, :] = a[2, :, :] @ b[:, :]
        For more info, see:
            https://numpy.org/doc/2.3/reference/generated/numpy.matmul.html
        Args:
            a (NDArray): ndarray with shape=[..., n, m]
            b (NDArray): ndarray with shape=[..., m, k]

        Returns:
            NDArray: output with shape=[..., n, k].
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        # Important: be sure to correctly support broadcast semantics (see `compute()` docstring)!
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def matmul(a: NDArray, b: NDArray) -> NDArray:
    return MatMul()(a, b)


class Negate(TensorOp):
    """Elementwise negation of input ndarray."""
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def negate(a: NDArray) -> NDArray:
    return Negate()(a)


class Log(TensorOp):
    """Element-wise natural log."""
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a: NDArray) -> NDArray:
    return Log()(a)


class Exp(TensorOp):
    """Calculate e^x element-wise."""
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a: NDArray) -> NDArray:
    return Exp()(a)


class ReLU(TensorOp):
    """Calculate relu(x) = max(0, x), elementwise."""
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


