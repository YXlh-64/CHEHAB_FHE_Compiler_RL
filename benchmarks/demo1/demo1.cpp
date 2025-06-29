#include "fheco/fheco.hpp"

using namespace std;
using namespace fheco;
#include <chrono>
#include <fstream>
#include <iostream> 
#include <string> 
#include <vector>
#include <cmath>
#include "../global_variables.hpp" 
/**************************/

void fhe_vectorized(int slot_count)
{ 
  Ciphertext a("a"); 
  Ciphertext b("b"); 
  Ciphertext c("c"); 
  Ciphertext d("d"); 
  Ciphertext e("e");  
  Ciphertext f("f");

  Ciphertext g("g");  
  Ciphertext h("h"); 
  Ciphertext output;

  output = ( (a + b) * (c + d) ) + ((e + f) * (g + h));
  
  output.set_output("output");
}

/************************************/
void fhe(int width)
{ 
  Ciphertext a("a"); 
  Ciphertext b("b"); 
  Ciphertext c("c"); 
  Ciphertext d("d"); 
  Ciphertext e("e");  
  Ciphertext f("f");

  Ciphertext g("g");  
  Ciphertext h("h"); 
  Ciphertext output;

  output = (  ( g + h) * (e + f) ) + ((c + d) * (a + b));
  
  output.set_output("output");
}
/******************************************************************************************/
/******************************************************************************************/
void print_bool_arg(bool arg, const string &name, ostream &os)
{
  os << (arg ? name : "no_" + name);
}
int main(int argc, char **argv)
{
  bool vectorize_code = true;
  if (argc > 1)
    vectorize_code = stoi(argv[1]);

  int window = 0;
  if (argc > 2) 
    window = stoi(argv[2]);

  bool call_quantifier = true;
  if (argc > 3)
    call_quantifier = stoi(argv[3]);

  bool cse = true; 
  if (argc > 4)
    cse = stoi(argv[4]);
   
  int slot_count = 1 ;
  if (argc > 5)
    slot_count = stoi(argv[5]);

  bool const_folding = true; 
  if (argc > 6)
    const_folding = stoi(argv[6]); 

  if (cse)
  {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
  }
  else
  {
    Compiler::disable_cse();
    Compiler::disable_order_operands();
  }

  if (const_folding)
    Compiler::enable_const_folding();
  else 
    Compiler::disable_const_folding(); 
  //Compiler::enable_auto_enc_params_selection();
  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  string func_name = "fhe";
  /**************/t = chrono::high_resolution_clock::now();
  if (vectorize_code)
  {
      const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
      fhe(slot_count);
      string gen_name = "_gen_he_" + func_name;
      string gen_path = "he/" + gen_name;
      ofstream header_os(gen_path + ".hpp");
      if (!header_os)
        throw logic_error("failed to create header file");
      ofstream source_os(gen_path + ".cpp");
      if (!source_os) 
        throw logic_error("failed to create source file");
      cout << " window is " << window << endl;
      Compiler::gen_vectorized_code(func, window);
      auto ruleset = Compiler::Ruleset::depth;
      auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
      Compiler::compile(func, ruleset, rewrite_heuristic, header_os, gen_name + ".hpp", source_os);
      Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os, 29);
      /************/elapsed = chrono::high_resolution_clock::now() - t;
      cout << elapsed.count() << " ms\n";
      if (call_quantifier) 
      {
        util::Quantifier quantifier{func};
        quantifier.run_all_analysis();
        quantifier.print_info(cout);
      }
  }
  else
  {
      const auto &func = Compiler::create_func(func_name,slot_count*slot_count, 20, false, true);
      // update io file 
      std::string updated_inputs_file_name = "fhe_io_example_adapted.txt" ;
      std::string inputs_file_name = "fhe_io_example.txt";
      util::copyFile(inputs_file_name,updated_inputs_file_name);
      fhe(slot_count);
      string gen_name = "_gen_he_" + func_name;
      string gen_path = "he/" + gen_name;
      ofstream header_os(gen_path + ".hpp");
      if (!header_os)
        throw logic_error("failed to create header file");
      ofstream source_os(gen_path + ".cpp");
      if (!source_os) 
        throw logic_error("failed to create source file");
      cout << " window is " << window << endl; 
      auto ruleset = Compiler::Ruleset::simplification_ruleset;
      auto rewrite_heuristic = trs::RewriteHeuristic::bottom_up;
      Compiler::compile(func, ruleset, rewrite_heuristic, header_os, gen_name + ".hpp", source_os);
      Compiler::gen_he_code(func, header_os, gen_name + ".hpp", source_os, 29);
      /************/elapsed = chrono::high_resolution_clock::now() - t;
      cout << elapsed.count() << " ms\n";
      if (call_quantifier)
      {
        util::Quantifier quantifier{func};
        quantifier.run_all_analysis();
        quantifier.print_info(cout);
      }
  }
  return 0;
}