
#if defined( __USE_HIP__ )
#include <tutorials/common/HipTestBase.h>
#include <cstdlib>
#include <numeric>

class Test : public HipTestBase
{

  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtError e = hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );
		ASSERT( e == hiprtSuccess );
		printf( "Create hip-hiprt context\n" );
	}
};
#endif

int main( int argc, char** argv )
{
#if defined( __USE_HIP__ )
	Test test;
	test.init( 0 );
	test.run();
#endif 
	return 0;
}

