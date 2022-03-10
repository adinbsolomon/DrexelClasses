public class CalculatorVisitor implements Visitor {

	@Override
	public int visit(Number number) {
		return number.getValue();
	}

	@Override
	public int visit(Addition addition) {
		return addition.getLeft().accept(this) + addition.getRight().accept(this);
	}

	@Override
	public int visit(Subtraction subtraction) {
		return subtraction.getLeft().accept(this) - subtraction.getRight().accept(this);
	}

}
