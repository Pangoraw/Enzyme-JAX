import jax
import jax.numpy as jnp
from enzymead.jax import enzyme_jax_ir
from absl.testing import absltest
import timeit


@enzyme_jax_ir()
def add_one(x: jax.Array, y) -> jax.Array:
    return x + 1 + y


@jax.jit
def add_one_plain(x: jax.Array, y) -> jax.Array:
    return x + 1 + y


@enzyme_jax_ir()
def add_two(x: jax.Array, z, y) -> jax.Array:
    return x + y


@jax.jit
def add_two_plain(x: jax.Array, z, y) -> jax.Array:
    return x + y


@jax.jit
def fwd(in0, in1, din0, din1):
    return jax.jvp(add_one, (in0, in1), (din0, din1))


@jax.jit
def fwd_plain(in0, in1, din0, din1):
    return jax.jvp(add_one_plain, (in0, in1), (din0, din1))


@jax.jit
def fwd2(in0, in1, in2, din0, din1, din2):
    return jax.jvp(add_two, (in0, in1, in2), (din0, din1, din2))


@jax.jit
def fwd2_plain(in0, in1, in2, din0, din1, din2):
    return jax.jvp(add_two_plain, (in0, in1, in2), (din0, din1, din2))


@jax.jit
def rev(in0, in1, dout):
    primals, f_vjp = jax.vjp(add_one, in0, in1)
    grads = f_vjp(dout)
    return primals, grads


@jax.jit
def rev_plain(in0, in1, dout):
    primals, f_vjp = jax.vjp(add_one_plain, in0, in1)
    grads = f_vjp(dout)
    return primals, grads


@jax.jit
def rev2(in0, in1, in2, dout):
    primals, f_vjp = jax.vjp(add_two, in0, in1, in2)
    grads = f_vjp(dout)
    return primals, grads


@jax.jit
def rev2_plain(in0, in1, in2, dout):
    primals, f_vjp = jax.vjp(add_two_plain, in0, in1, in2)
    grads = f_vjp(dout)
    return primals, grads


class AddOneTwo(absltest.TestCase):
    def setUp(self):
        self.in0 = jnp.array([1.0, 2.0, 3.0])
        self.in1 = jnp.array([10.0, 20.0, 30.0])
        self.in2 = jnp.array([100.0, 200.0, 300.0])
        self.din0 = jnp.array([0.1, 0.2, 0.3])
        self.din1 = jnp.array([50.0, 70.0, 110.0])
        self.din2 = jnp.array([1300.0, 1700.0, 1900.0])

    def test_add_one_primal(self):
        ao = add_one(self.in0, self.in1)
        aop = add_one_plain(self.in0, self.in1)
        self.assertTrue((jnp.abs(ao - aop) < 1e-6).all())

        # Benchmark.
        print(
            timeit.Timer(
                "add_one(in0, in1)",
                globals={"add_one": add_one, "in0": self.in0, "in1": self.in1},
            ).timeit()
        )
        print(
            timeit.Timer(
                "add_one_plain(in0, in1)",
                globals={
                    "add_one_plain": add_one_plain,
                    "in0": self.in0,
                    "in1": self.in1,
                },
            ).timeit()
        )

    def test_add_two_deadarg(self):
        at = add_two(self.in0, self.in1, self.in2)
        atp = add_two_plain(self.in0, self.in1, self.in2)
        self.assertTrue((jnp.abs(at - atp) < 1e-6).all())

    def test_add_one_forward(self):
        primals, tangents = fwd(self.in0, self.in1, self.din0, self.din1)
        primals_p, tangents_p = fwd_plain(self.in0, self.in1, self.din0, self.din1)

        self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())
        for t, t_p in zip(tangents, tangents_p):
            self.assertTrue((jnp.abs(t - t_p) < 1e-6).all())

        print(
            timeit.Timer(
                "fwd(in0, in1, din0, din1)",
                globals={
                    "fwd": fwd,
                    "in0": self.in0,
                    "in1": self.in1,
                    "din0": self.din0,
                    "din1": self.din1,
                },
            ).timeit()
        )
        print(
            timeit.Timer(
                "fwd_plain(in0, in1, din0, din1)",
                globals={
                    "fwd_plain": fwd_plain,
                    "in0": self.in0,
                    "in1": self.in1,
                    "din0": self.din0,
                    "din1": self.din1,
                },
            ).timeit()
        )

    def test_add_two_deadarg_forward(self):
        primals, tangents = fwd2(
            self.in0, self.in1, self.in2, self.din0, self.din1, self.din2
        )
        primals_p, tangents_p = fwd2_plain(
            self.in0, self.in1, self.in2, self.din0, self.din1, self.din2
        )

        print(primals, primals_p)
        self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())
        for i, (t, t_p) in enumerate(zip(tangents, tangents_p)):
            print(i, t, t_p)
            self.assertTrue((jnp.abs(t - t_p) < 1e-6).all())

    def test_add_one_reverse(self):
        dout = jnp.array([500.0, 700.0, 110.0])

        primals, grads = rev(self.in0, self.in1, dout)
        # TODO enzyme will in place 0 the gradient inputs, which may not be expected
        print(dout)
        dout = jnp.array([500.0, 700.0, 110.0])
        primals_p, grads_p = rev_plain(self.in0, self.in1, dout)

        self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())
        for i, (g, g_p) in enumerate(zip(grads, grads_p)):
            print(i, g, g_p)
            self.assertTrue((jnp.abs(g - g_p) < 1e-6).all())

        print(
            timeit.Timer(
                "rev(in0, in1, dout)",
                globals={"rev": rev, "in0": self.in0, "in1": self.in1, "dout": dout},
            ).timeit()
        )
        print(
            timeit.Timer(
                "rev_plain(in0, in1, dout)",
                globals={
                    "rev_plain": rev_plain,
                    "in0": self.in0,
                    "in1": self.in1,
                    "dout": dout,
                },
            ).timeit()
        )

    def test_add_two_deadarg_reverse(self):
        dout = jnp.array([500.0, 700.0, 110.0])
        primals, grads = rev2(self.in0, self.in1, self.in2, dout)
        # TODO enzyme will in place 0 the gradient inputs, which may not be expected
        print(dout)
        dout = jnp.array([500.0, 700.0, 110.0])
        primals_p, grads_p = rev2_plain(self.in0, self.in1, self.in2, dout)

        self.assertTrue((jnp.abs(primals - primals_p) < 1e-6).all())
        for i, (g, g_p) in enumerate(zip(grads, grads_p)):
            print(i, g, g_p)
            self.assertTrue((jnp.abs(g - g_p) < 1e-6).all())


@enzyme_jax_ir()
def esum(x):
    return jnp.sum(x)


@jax.jit
def sumfwd(in0, din0):
    return jax.jvp(esum, (in0,), (din0,))


@jax.jit
def sumrev_p(in0):
    primals, f_vjp = jax.vjp(jnp.sum, in0)
    grads = f_vjp(1.0)
    return primals, grads


@jax.jit
def sumrev(in0):
    primals, f_vjp = jax.vjp(esum, in0)
    grads = f_vjp(1.0)
    return primals, grads


class Sum(absltest.TestCase):
    def setUp(self):
        self.x = jnp.array(range(50), dtype=jnp.float32)
        self.dx = jnp.array([i * i for i in range(50)], dtype=jnp.float32)

    def test_primal(self):
        eres = esum(self.x)
        print(eres)
        self.assertTrue(jnp.abs(eres - 50 * 49 / 2) < 1e-6)

    def test_forward(self):
        primals, tangents = sumfwd(self.x, self.dx)
        print(primals, tangents)
        self.assertTrue(jnp.abs(primals - 50 * 49 / 2) < 1e-6)
        self.assertTrue(jnp.abs(tangents - 50 * 49 * 99 / 6) < 1e-6)

    def test_reverse_p(self):
        primals, grads = sumrev_p(self.x)
        print(primals, grads)

    def test_reverse(self):
        primals, grads = sumrev(self.x)
        print(primals, grads)
        self.assertTrue(jnp.abs(primals - 50 * 49 / 2) < 1e-6)
        self.assertTrue((jnp.abs(grads[0] - 1) < 1e-6).all())


@enzyme_jax_ir()
def ecache(x):
    return x * x[0]


@jax.jit
def cacherev(in0, din0):
    primals, f_vjp = jax.vjp(ecache, in0)
    grads = f_vjp(din0)
    return grads


class Cache(absltest.TestCase):
    def test_reverse(self):
        dim = 288

        x = jnp.array(range(dim), dtype=jnp.float32)
        dx = jnp.array(range(dim), dtype=jnp.float32)

        grads = cacherev(x, dx)
        self.assertTrue(
            jnp.abs(grads[0][0] - (dim - 1) * dim * (2 * (dim - 1) + 1) / 6) < 1e-6
        )
        self.assertTrue((jnp.abs(grads[0][1:]) < 1e-6).all())


if __name__ == "__main__":
    absltest.main()
