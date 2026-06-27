// demo.c —— %dead 那行计算结果从未被使用，是典型死代码
int compute(int x) {
  int dead = x * 12345;   // 计算后从未使用 => 死代码
  int useful = x + 1;
  return useful;          // 只用到 useful
}
