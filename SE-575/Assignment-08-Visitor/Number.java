public class Number extends Component {

	protected int value;

	public Number(int value) {
		this.value = value;
	}

	public int getValue() {
		return this.value;
	}

	@Override
	public int accept(Visitor visitor) {
		return visitor.visit(this);
	}

}
