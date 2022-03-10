public abstract class BinaryComponent extends Component {

	protected Component left;
	protected Component right;

	public BinaryComponent(Component left, Component right) {
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
