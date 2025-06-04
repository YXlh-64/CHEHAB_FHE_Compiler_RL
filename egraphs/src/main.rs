#![allow(warnings)]
extern crate clap;
use clap::{App, Arg};
use egraphslib::*;
use std::time::Instant; 
use crate::veclang::VecLang;
use egg::{RecExpr, Id, Language};
use rand::Rng;
use std::{env, fs};
/************************************/
use std::collections::BinaryHeap;
use std::cmp::Ordering; 
use std::cmp::Reverse; 

#[derive(Debug, Eq, PartialEq)]
struct State {
    cost: usize,
    max_vector_width : usize ,
    expression: RecExpr<VecLang>, // Additional data (e.g., node ID)
}

// Implement Ord for custom sorting based on cost
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // Min-heap: lower cost has higher priority
    }
}

// Implement PartialOrd to be consistent with Ord
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/************************************/
fn main() { 
    /******************************************
    let matches = App::new("Rewriter")
    .arg(
        Arg::with_name("INPUT")
            .help("Sets the input file") 
            .required(true)
            .index(1), 
    ) 
    .arg(
        Arg::with_name("vector_width")
            .help("Sets the vector_width")
            .required(true)
            .index(2),
    ) 
    .get_matches();
    // Get a path string to parse a program.
    let path = matches.value_of("INPUT").unwrap();
    let timeout = env::var("TIMEOUT")
        .ok()
        .and_then(|t| t.parse::<u64>().ok())
        .unwrap_or(300);
    let prog_str = fs::read_to_string(path).expect("Failed to read the input file.");
    let mut prog : RecExpr<VecLang>= prog_str.parse().unwrap();
    let vector_width: usize = matches
        .value_of("vector_width")
        .unwrap() 
        .parse()
        .expect("Number must be a valid usize");
    // Push elements with different costs
    let node_limit = 100_000 ; 
    let start_time = Instant::now();
    let beam_width : usize = 4 ;
    let mut priorityQueue = BinaryHeap::new();
    let mut temp_priority_Queue = BinaryHeap::new();
    let initial_expression = prog.clone();
    let mut best_expr = prog.clone();
    let mut best_cost = usize::MAX;
    let mut current_vector_width  = vector_width ;
    priorityQueue.push(State {expression: initial_expression , max_vector_width: current_vector_width ,cost: usize::MAX });
    let rulesets_appplying_order  = vec![2,3,4,5];
    for i in 0..20{ 
        while let Some(state) = priorityQueue.pop() {
            for iteration in 0..4 {
                let (expression_cost, updated_expression, stop_reason) = rules::run(&state.expression, timeout, state.max_vector_width,node_limit,rulesets_appplying_order[iteration%rulesets_appplying_order.len()],0);
                // evaluate depth of the new expression : 
                current_vector_width = rules::get_vector_width(&updated_expression);
                temp_priority_Queue.push(State{expression: updated_expression,max_vector_width: current_vector_width,cost: expression_cost})
            }
        }
        /*************************************************/
        let mut index = 0;
        while let Some(state) = temp_priority_Queue.pop(){
            if index >= beam_width {
                break; 
            }
            eprintln!("\n =========> best reached cost : {} \n",state.cost);
            priorityQueue.push(state);
            index+=1;
        }
    }
    let duration = start_time.elapsed();
    if let Some(state) = priorityQueue.pop(){
        best_expr = state.expression ;
        best_cost = state.cost ;
        current_vector_width = state.max_vector_width ;
    }
    println!("{}", best_expr.to_string()); /* Pretty print with width 80 */
    println!("{} {}",current_vector_width,current_vector_width);
    eprintln!("\n===> Final expression depth : {}", rules::ast_depth(&best_expr));
    eprintln!("\nCost: {}", best_cost);
    eprintln!("Time taken: {:?} to finish", duration);
    ****************************************************************/



    /***************************************************************/
    let matches = App::new("Rewriter")
    .arg(
        Arg::with_name("INPUT")
            .help("Sets the input file") 
            .required(true)
            .index(1), 
    ) 
    .arg(
        Arg::with_name("vector_width")
            .help("Sets the vector_width")
            .required(true)
            .index(2),
    ) 
    .get_matches();
    // Get a path string to parse a program.
    let path = matches.value_of("INPUT").unwrap();
    let timeout = env::var("TIMEOUT")
        .ok()
        .and_then(|t| t.parse::<u64>().ok())
        .unwrap_or(300);
    let prog_str = fs::read_to_string(path).expect("Failed to read the input file.");
    let mut prog : RecExpr<VecLang>= prog_str.parse().unwrap();
    let vector_width: usize = matches
        .value_of("vector_width")
        .unwrap() 
        .parse()
        .expect("Number must be a valid usize");
    // Push elements with different costs
    let node_limit = 100_000 ; 
    let start_time = Instant::now();
    // Record the start time
    let start_time = Instant::now();

    // Run rewriter
    eprintln!(
        "Running egg with timeout {:?}s, width: {:?}",
        timeout, vector_width
    );
    /*********************************/
    let mut current_cost = 0 ;
    let mut current_expr = prog.clone();
    let mut stop_reason = 0 ;
    let mut rulesets_appplying_order  = vec![0,1];
    let mut node_limit = 100 ;
    /*********************************/ 
    rulesets_appplying_order  = vec![2];
    let mut previous_cost = usize::MAX;
    node_limit = 100_000 ;  
    let mut comp = 0;
    let mut iteration = 0;
    let mut current_vector_width = vector_width ; 
    while (comp != rulesets_appplying_order.len()){
        let (cost, best, stop_reason) = rules::run(&current_expr, timeout, current_vector_width,node_limit,rulesets_appplying_order[iteration%rulesets_appplying_order.len()],0);
        current_expr = best ; 
        current_cost = cost ;
        current_vector_width = rules::get_vector_width(&current_expr);
        if (current_cost == previous_cost){
            comp+=1;
        }else{ 
            previous_cost=current_cost ; 
            comp=0;
        }
        iteration = iteration + 1 ;
        eprintln!("Best cost at iteration {}: {} ", iteration + 1, current_cost);
        //eprintln!("Obtained expression ==> : {}", current_expr.to_string());
    }
    let mut best_cost = current_cost ;
    let mut best_expr = current_expr.clone(); 
    /*****************************************************************************
    rulesets_appplying_order  = vec![9,10,11,12,13,14,15];
    let mut best_depth = rules::ast_depth(&best_expr);
    for iteration in 0..24 {
        let (cost, best , stop_reason) = rules::run(&current_expr, timeout, vector_width, node_limit,rulesets_appplying_order[iteration%rulesets_appplying_order.len()],0);
        let depth = rules::ast_depth(&best);
        if depth < best_depth {
            best_depth=depth ;  
            current_expr = best ; 
            best_expr = current_expr.clone() ;
        }
        eprintln!("Best cost at iteration {}: {} , Depth {} \n\n", iteration + 1, cost,depth);
    }
    *****************************************************************************/
    let duration = start_time.elapsed();
    // Print the results
    println!("{}", best_expr.to_string()); /* Pretty print with width 80 */
    println!("{} {}",current_vector_width,current_vector_width);
    eprintln!("\n===> Final expression depth : {}", rules::ast_depth(&best_expr));
    eprintln!("\nCost: {}", best_cost);
    //eprintln!("Time taken: {:?} to finish", duration);*/
}
