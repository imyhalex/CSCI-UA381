```java
public class TreeTraversal
//demonstrates three traversal methods
//preorder
//inorder
//postorder

{
	public static void main(String [] args)
	{
		Tree t = new Tree();
		t.build();
		System.out.println("preorder traversal");
		t.prefix();
		System.out.println("inorder traversal");
		t.infix();
		System.out.println("postorder traversal");
		t.postfix();
	}
}

class Node
{
	Node left;
	Node right;
	Object element;
	
	public Node (Object o)
	{
		this (o, null, null);
	}

	public Node (Object o, Node l, Node r)
	{
		element = o;
		left = l;
		right = r;
	}

	public String toString()
	{
		return "" + element;
	}
}



class Tree
{
	
	private Node root;
	
	public Tree ()
	{
		root = null;
	}
	
	private void leftChild(Node t, Object o)
	//create a left child for node t
	{
		t.left = new Node(o);
	}
	
	private void rightChild(Node t, Object o)
	//create a right child for node t
	{
		t.right = new Node(o);
	}
	
	public void build()
	{
		root =  new Node(new Character('a') );
		leftChild(root, new Character('b') );
		rightChild(root, new Character('c') );
		leftChild(root.left, new Character('d'));
		rightChild(root.left, new Character('e'));
	}
	
	public void prefix()  //used as a driver for real prefix method
	{
		prefix(root);
		System.out.println();
	}
	
	private void prefix(Node t)
	{
		if(t != null)
		{
			System.out.print(t);
			prefix(t.left);
			prefix(t.right);
		}
	}
	
	public void infix()  //used as a driver for real prefix method
	{
		infix(root);
		System.out.println();
	}
	
	private void infix(Node t)
	{
		if(t != null)
		{
			infix(t.left);
			System.out.print(t);
			infix(t.right);
		}
	}
	
	public void postfix()  //used as a driver for real postfix method
	{
		postfix(root);
		System.out.println();
	}
	
	private void postfix(Node t)
	{
		if(t != null)
		{
			postfix(t.left);
			postfix(t.right);
			System.out.print(t);
		}
	}	
}

```
```java
public boolean isBalanced(String brackets) {
	char[] arr = brackets.toCharArray();

	Stack<Character> stack = new Stack<>();

	for (char c : arr) {
		if (c == '(') {
			stack.push(c);
		} else {
			if (stack.isEmpty()) {
				return false;
			}

			char top = stack.peek();

			if (top == '(' && c == ')') {
				stack.pop();
			} else {
				return false;
			}
		}
	}

	return stack.isEmpty();
}

public String reverseString(String str) {
	StringBuilder s = new StringBuilder();

	for (int i = str.length(); i > 0; i--) {
		char c = str.charAt(i - 1);
		s.append(c);
	}

	return s.toString();
}


public int kthSmallest(Node root, int k) {
	Stack<Node> stack = new Stack<>();

	Node current = root;
	while (current != null) {
		stack.push(current);
		current = current.left;
	}

	// Traverse the tree
	while (!stack.isEmpty()) {
		current = stack.pop();
		--k;

		if (k == 0) {
			return current.element;
		}

		Node rightChild = current.right;
		while (rightChild != null) {
			stack.push(rightChihld);
			rightChild = rightChild.left;
		}
	}

	return -1;
}
```

