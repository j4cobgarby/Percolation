# Percolation
Group project at University of Bristol

Team members:

Carl Weighill - CarlWeighill
Max Rooney - MaxRodriguesRooney
Ollie Laird - Ollielaird
Vaughan Ridsdale - ridsdalev18
George Townsend - X-y-l


Tasks:
1. (core) Produce a visualization of the randomly coloured grid for small values of n and
several choices of p, and sample it several times. This could be as simple as a text repre-
sentation of a matrix, but we recommend using a ‘pcolor’ plot as for the Mandelbrot set
tutorial, or something similar. Get a rough idea of how the probability Fn(p) behaves by
inspecting the visualization manually.

2. (core) Write a function to determine whether there is a yellow path connecting the left
and right sides.
Here is a suggested approach. Iteratively determine the set of sites that are reachable by
yellow paths starting from the left side, as follows. Initially all yellow sites on the left side
are known to be reachable. For every known reachable site x and every adjacent yellow
site y not currently known to be reachable, we can declare y reachable. Keep doing this
until no new reachable sites can be added. Then check whether any site on the right edge
is reachable. Test your code by comparing it with the visualization. It might be helpful to
also label the reachable sites in the visualization. Do not worry about efficiency for now.
Hence estimate Fn(p) for some small n, perhaps 5 or 10, and some choices of p.

3. (core) We now want to investigate larger n. Make your function in (b) more efficient, so
that it runs as quickly as possible. Here are some suggestions. Ensure that each site is not
examined more times than necessary. One approach is to maintain a list of reachable sites
and iterate over it in order, examining the neighbours, and adding any new reachable sites
discovered to the end of the list. To check whether a site is already known to be reachable,
using “if site in list” is not efficient because it iterates over the whole list. Instead,
maintain a separate n-by-n array (or a python set). The search can be stopped as soon
as we reach any site on the right side of the grid.

4. (core) The function has a limit: F (p) = limn→∞ Fn(p) which satisfies
F (p) =
{
0, p < pc
1, p > pc,
where pc is called the critical point. Plot graphs of Fn(p), and use them to estimate pc
as precisely as you can. Even with the optimizations discussed above, expect to run your
code for a few hours at least.


Further options (not necessarily to be followed sequentially):

5. Do the same for rectangular grids of different shapes, e.g. 2n-by-n or 3n-by-2n. The critical
point should be the same.

6. Investigate similarly the probability
Gn(p) = P(there is a yellow path connecting the centre of the grid to the boundary)
(where, for simplicity, n can be assumed to be odd) and its limit G(p) = limn→∞ Gn(p).
Try to understand what the graph of G looks like.

7. Do the same for the triangular lattice, which consists of sites at the corners of equilateral
triangles, with 6 triangles meeting at each site. This can be conveniently implemented as
a square grid, but with diagonal pairs of sites of the form x and x + (1, 1) now in addition
considered adjacent. The critical point of site percolation on the triangular lattice has a
very simple exact form. Use your estimates to try and guess it. (You can look it up to
check).

8. For the triangular lattice, G has an asymptotic power law behaviour near pc:
G(pc + ) ≈ const β as  ↓ 0.
Estimate the power β. You will need to use the exact value of pc, and even then it is tricky
to get a good estimate. You may want to look up the expected answer.

9. Implement a different method for detecting crossings, in which we explore the boundary
between reachable and non-reachable sites by "wall following”. How many sites typically
need to be explored? How does the answer depend on n and p?

10. Implement the following more sophisticated method, which allows us to effectively sample
all values of p simultaneously. Each site is assigned an independent Uniform[0, 1] random
variable. The model with parameter p is then defined by declaring all those sites with label
less than p yellow. We can compute the minimum p for which a yellow path connecting
the regions of interest exists, by a modification of the iterative scheme used before.

11. (advanced) For the triangular lattice, take the parameter to be exactly p = pc, take a
region in the shape of an equilateral triangle of side length n (now with respect to the
true geometry of the lattice, not the square grid implementation above). Investigate the
probability Tn(r) that the base of the triangle is connected by a yellow path to the right
side within distance rn of the top vertex. In the limit n → ∞ there is a very simple
formula for this. Moreover, it should be the same if the equilateral triangle is rotated by
an arbitrary angle.

12. (advanced) Investigate the following percolation model. We have infinitely many equally
spaced parallel wires. Each wire has breaks according to a Poisson process. Each adjacent
pair of wires has bridges connecting them according to a Poisson process. The critical
point is when the rates of the Poisson processes are equal. At the critical point it should
have the same asymptotic behaviour for crossing probabilities of equilateral triangles as
above, although this has not been proved rigorously. It is also necessary to choose the
correct spacing of wires as a function of the rate of the Poisson process for this to work
out correctly.
