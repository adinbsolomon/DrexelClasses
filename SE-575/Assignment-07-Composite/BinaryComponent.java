public abstract class BinaryComponent extends Component {

	protected Component left;
	protected Component right;

	public BinaryComponent(String name, Component left, Component right) {
		super(name);
		this.left = left;
		this.right = right;
	}

	public Component getLeft() {
		return left;
	}

	public Component getRight() {
		return right;
	}
}
