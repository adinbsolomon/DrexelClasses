public class Addition extends BinaryComponent {

	public Addition(Component left, Component right) {
		super(left, right);
	}

	@Override
	public int accept(Visitor visitor) {
		return visitor.visit(this);
	}

}
