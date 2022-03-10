public class Number extends Component {

	public static String NUMBER_NAME = "number";

	protected int value;

	public Number(int value) {
		super(NUMBER_NAME);
		this.value = value;
	}

	public int getValue() {
		return this.value;
	}

}
