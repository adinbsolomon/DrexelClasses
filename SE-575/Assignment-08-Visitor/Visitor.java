public interface Visitor {

	public int visit(Number number);

	public int visit(Addition addition);

	public int visit(Subtraction subtraction);

}
