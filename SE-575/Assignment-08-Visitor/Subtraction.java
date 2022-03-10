public class Subtraction extends BinaryComponent {

	public Subtraction(Component left, Component right) {
		super(left, right);
	}

	@Override
	public int accept(Visitor visitor) {
		return visitor.visit(this);
	}

}
