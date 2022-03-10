public abstract class Component {

	protected String name;

	protected Component(String name) {
		this.name = name;
	}

	public String getType() {
		return this.name;
	}

}
